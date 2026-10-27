"""Focused tests consolidated by validation surface."""

import json
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import httpx

import publication_status
import metadata_providers
import publication_audit
import record_matching


# Commentary interpretation

class CommentaryAuditTests(unittest.TestCase):
    def test_pubmed_comment_links_and_direction(self) -> None:
        with self.subTest(case="pubmed_linked_comment_is_reported"):
            xml = """
        <PubmedArticleSet><PubmedArticle><MedlineCitation>
          <CommentsCorrectionsList>
            <CommentsCorrections RefType="CommentIn"><PMID>123</PMID></CommentsCorrections>
            <CommentsCorrections RefType="ErratumIn"><PMID>456</PMID></CommentsCorrections>
          </CommentsCorrectionsList>
        </MedlineCitation></PubmedArticle></PubmedArticleSet>
        """

            assessment = publication_status.assess_commentary(
                pubmed_xml=xml,
                pubmed_checked=True,
                pubmed_expected=True,
            )

            self.assertEqual(assessment.status, "linked_comment_found")
            self.assertEqual(
                tuple(link.report_text() for link in assessment.pubmed_comment_links),
                ("CommentIn:PMID:123",),
            )

        with self.subTest(case="pubmed_comment_direction_is_preserved"):
            xml = """
        <PubmedArticleSet><PubmedArticle><MedlineCitation>
          <CommentsCorrectionsList>
            <CommentsCorrections RefType="CommentOn"><PMID>321</PMID></CommentsCorrections>
          </CommentsCorrectionsList>
        </MedlineCitation></PubmedArticle></PubmedArticleSet>
        """

            assessment = publication_status.assess_commentary(
                pubmed_xml=xml,
                pubmed_checked=True,
                pubmed_expected=True,
            )

            self.assertEqual(
                assessment.pubmed_comment_links[0].report_text(),
                "CommentOn:PMID:321",
            )


    def test_scite_tally_interpretation(self) -> None:
        with self.subTest(case="contradicting_scite_tally_has_specific_status"):
            assessment = publication_status.assess_commentary(
                scite_payload={
                    "total": 42,
                    "supporting": 30,
                    "contradicting": 2,
                    "mentioning": 10,
                    "unclassified": 0,
                    "citingPublications": 38,
                },
                scite_checked=True,
                scite_expected=True,
            )

            self.assertEqual(assessment.status, "critical_citation_found")
            self.assertEqual(assessment.scite.contradicting, 2)
            self.assertIn("not retraction", assessment.note)

        with self.subTest(case="noncritical_tally_is_citation_context"):
            assessment = publication_status.assess_commentary(
                scite_payload={"total": 4, "supporting": 1, "mentioning": 3},
                scite_checked=True,
                scite_expected=True,
            )

            self.assertEqual(assessment.status, "citation_context_found")

        with self.subTest(case="checked_zero_tally_has_no_commentary_signal"):
            assessment = publication_status.assess_commentary(
                scite_payload={"total": 0},
                scite_checked=True,
                scite_expected=True,
            )

            self.assertEqual(assessment.status, "no_commentary_signal")

        with self.subTest(case="unavailable_expected_scite_tally_is_incomplete"):
            assessment = publication_status.assess_commentary(scite_expected=True)

            self.assertEqual(assessment.status, "check_incomplete")
            self.assertIn("Scite tally was unavailable", assessment.note)

        with self.subTest(case="invalid_scite_counts_are_not_coerced_to_findings"):
            tallies = publication_status.parse_scite_tallies(
                {"total": -1, "contradicting": True, "mentioning": "unknown"}
            )

            self.assertIsNone(tallies.total)
            self.assertIsNone(tallies.contradicting)
            self.assertIsNone(tallies.mentioning)


# Metadata provider adapters

def _provider_record(provider: str) -> record_matching.ProviderRecord:
    author = record_matching.Author(given="A", family="Example")
    meta = record_matching.PaperMeta(
        title="Reliable Quantitative Imaging with Magnetic Resonance",
        year=2024,
        journal_abbrev="Magn Reson Med",
        first_author=author,
        authors=[author],
        corresponding_author=author,
        doi="10.1000/reliable",
        pmid="123" if provider == "pubmed" else None,
        author_keywords=[],
        mesh_keywords=[],
    )
    return record_matching.ProviderRecord(
        provider,
        meta,
        "123" if provider == "pubmed" else None,
    )


class MetadataProviderSetTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_adapters_are_queried(self) -> None:
        with self.subTest(case="doi_lookup_queries_every_adapter_in_provider_order"):
            calls: list[str] = []

            class Adapter:
                def __init__(self, provider: str):
                    self.provider = provider

                async def records_by_doi(self, doi: str):
                    calls.append(f"{self.provider}:{doi}")
                    return metadata_providers.ProviderLookup(
                        (_provider_record(self.provider),),
                        (f"{self.provider}-note",),
                    )

            providers = metadata_providers.MetadataProviders(
                Adapter("pubmed"),
                Adapter("crossref"),
                Adapter("datacite"),
            )

            lookup = await providers.records_by_doi("10.1000/reliable")

            self.assertEqual(
                [record.provider for record in lookup.records],
                ["pubmed", "crossref", "datacite"],
            )
            self.assertEqual(
                calls,
                [
                    "pubmed:10.1000/reliable",
                    "crossref:10.1000/reliable",
                    "datacite:10.1000/reliable",
                ],
            )
            self.assertEqual(
                lookup.notes,
                ("pubmed-note", "crossref-note", "datacite-note"),
            )

        with self.subTest(case="title_lookup_queries_every_adapter"):
            calls: list[str] = []

            class Adapter:
                def __init__(self, provider: str):
                    self.provider = provider

                async def records_by_title(self, title: str):
                    calls.append(f"{self.provider}:{title}")
                    return metadata_providers.ProviderLookup(
                        (_provider_record(self.provider),)
                    )

            providers = metadata_providers.MetadataProviders(
                Adapter("pubmed"),
                Adapter("crossref"),
                Adapter("datacite"),
            )

            lookup = await providers.records_by_title("Reliable Imaging")

            self.assertEqual(len(lookup.records), 3)
            self.assertEqual(
                calls,
                [
                    "pubmed:Reliable Imaging",
                    "crossref:Reliable Imaging",
                    "datacite:Reliable Imaging",
                ],
            )


class PubMedAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_notice_lookup_does_not_substitute_original_article(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            closing(metadata_providers.MetadataCache(Path(directory) / "cache.sqlite")) as cache,
        ):
            cache.open()
            adapter = metadata_providers.PubMedAdapter(mock.Mock(), cache, "", "", mock.Mock(), {}, {})
            notice_xml = '''<PubmedArticleSet><PubmedArticle><MedlineCitation>
              <PMID>200</PMID><Article><ArticleTitle>Retraction Note: Synthetic study</ArticleTitle>
              <PublicationTypeList><PublicationType>Retraction of Publication</PublicationType></PublicationTypeList>
              <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
              <AuthorList><Author><LastName>Example</LastName><ForeName>A</ForeName></Author></AuthorList></Article>
              <CommentsCorrectionsList><CommentsCorrections RefType="RetractionOf"><PMID>100</PMID></CommentsCorrections></CommentsCorrectionsList>
              </MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.5555/notice.0001</ArticleId></ArticleIdList></PubmedData>
              </PubmedArticle></PubmedArticleSet>'''
            with mock.patch.object(metadata_providers, "fetch_http", mock.AsyncMock(return_value=notice_xml)) as fetch:
                meta, pmid = await adapter.fetch_meta("200")
            self.assertEqual(pmid, "200")
            self.assertEqual(meta.doi, "10.5555/notice.0001")
            fetch.assert_awaited_once()

    async def test_mismatched_cached_pmid_is_discarded(self) -> None:
        requested_doi = "10.1109/TMTT.2018.2869603"
        with tempfile.TemporaryDirectory() as directory:
            cache = metadata_providers.MetadataCache(
                Path(directory) / "metadata.sqlite"
            )
            cache.open()
            try:
                cache.put_doi2pmid(requested_doi, "41116384")
                adapter = metadata_providers.PubMedAdapter(
                    mock.Mock(),
                    cache,
                    "",
                    "",
                    mock.Mock(),
                    {},
                    {},
                )
                unrelated = _provider_record("pubmed").meta
                unrelated.doi = "10.1103/lt46-nrh4"

                with (
                    mock.patch.object(
                        metadata_providers.PubMedAdapter,
                        "fetch_meta",
                        mock.AsyncMock(return_value=(unrelated, "41116384")),
                    ),
                    mock.patch(
                        "metadata_providers.fetch_http",
                        mock.AsyncMock(
                            return_value={
                                "records": [],
                                "esearchresult": {"idlist": []},
                            }
                        ),
                    ),
                ):
                    lookup = await adapter.records_by_doi(requested_doi)

                self.assertEqual(lookup.records, ())
                self.assertIsNone(cache.get_doi2pmid(requested_doi))
                self.assertTrue(
                    any("cached PubMed mapping" in note for note in lookup.notes)
                )
            finally:
                cache.close()


class CrossrefAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_alternative_id_lookup_normalizes_query_and_response(self) -> None:
        adapter = metadata_providers.CrossrefAdapter(
            mock.Mock(),
            mock.Mock(),
            mock.Mock(),
            {},
            mock.Mock(),
        )
        message = {
            "alternative-id": ["S 1352-8661 (00)00069-7"],
            "DOI": "10.1000/reliable",
            "title": ["Reliable Imaging"],
            "issued": {"date-parts": [[2000]]},
            "short-container-title": ["MAGMA"],
            "author": [{"given": "A", "family": "Hudson"}],
        }

        with mock.patch.object(
            metadata_providers.CrossrefAdapter,
            "messages_by_alternative_id",
            mock.AsyncMock(return_value=[message]),
        ) as lookup:
            result = await adapter.records_by_alternative_id(
                "S 1352-8661 (00)00069-7"
            )

        lookup.assert_awaited_once_with("S1352866100000697")
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].meta.doi, "10.1000/reliable")


class ProviderParserTests(unittest.TestCase):
    def test_provider_payload_normalization(self) -> None:
        with self.subTest(case="pubmed_xml_is_normalized"):
            xml = """
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>123</PMID>
              <Article>
                <ArticleTitle>Reliable Imaging</ArticleTitle>
                <Journal>
                  <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
                </Journal>
                <AuthorList>
                  <Author><LastName>Example</LastName><ForeName>A</ForeName></Author>
                </AuthorList>
              </Article>
              <MedlineJournalInfo><MedlineTA>Magn Reson Med</MedlineTA></MedlineJournalInfo>
            </MedlineCitation>
            <PubmedData>
              <ArticleIdList>
                <ArticleId IdType="doi">10.1000/reliable</ArticleId>
              </ArticleIdList>
            </PubmedData>
          </PubmedArticle>
        </PubmedArticleSet>
        """

            meta = metadata_providers.parse_pubmed_full(xml)

            self.assertIsNotNone(meta)
            assert meta is not None
            self.assertEqual(meta.pmid, "123")
            self.assertEqual(meta.doi, "10.1000/reliable")
            self.assertEqual(meta.first_author.family, "Example")

        with self.subTest(case="crossref_json_is_normalized"):
            meta = metadata_providers.parse_crossref_message(
                {
                    "DOI": "10.1000/reliable",
                    "title": ["Reliable Imaging"],
                    "issued": {"date-parts": [[2024]]},
                    "short-container-title": ["Magn Reson Med"],
                    "author": [{"given": "A", "family": "Example"}],
                }
            )

            self.assertIsNotNone(meta)
            assert meta is not None
            self.assertEqual(meta.year, 2024)
            self.assertEqual(meta.doi, "10.1000/reliable")
            self.assertEqual(meta.journal_abbrev, "Magn Reson Med")

        with self.subTest(case="datacite_json_is_normalized"):
            meta = metadata_providers.parse_datacite_attributes(
                {
                    "doi": "10.1000/reliable",
                    "titles": [{"title": "Reliable Imaging"}],
                    "publicationYear": 2024,
                    "creators": [
                        {"givenName": "A", "familyName": "Example"}
                    ],
                    "container": {"title": "Magn Reson Med"},
                }
            )

            self.assertIsNotNone(meta)
            assert meta is not None
            self.assertEqual(meta.year, 2024)
            self.assertEqual(meta.doi, "10.1000/reliable")
            self.assertEqual(meta.first_author.family, "Example")


# Publication audit orchestration

class FakeScite:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, ...]] = []

    async def fetch_many(self, dois):
        values = tuple(dois)
        self.calls.append(values)
        return {
            doi: publication_audit.SciteLookup(
                payload=dict(self.payload),
                checked=True,
            )
            for doi in values
        }


class AsyncPublicationAuditorTests(unittest.IsolatedAsyncioTestCase):
    async def test_audit_reuses_sources_and_lookup_payloads(self) -> None:
        with self.subTest(case="fresh_sources_are_combined_and_cached_for_one_run"):
            calls: list[str] = []
            scite = FakeScite({"total": 5, "contradicting": 1})
            pubmed_xml = """
        <PubmedArticleSet><PubmedArticle><MedlineCitation>
          <CommentsCorrectionsList>
            <CommentsCorrections RefType="RetractionIn"><PMID>200</PMID></CommentsCorrections>
            <CommentsCorrections RefType="CommentIn"><PMID>300</PMID></CommentsCorrections>
          </CommentsCorrectionsList>
        </MedlineCitation></PubmedArticle></PubmedArticleSet>
        """

            async def fetch(_client, url, _params, accept_xml, *_args):
                calls.append(url)
                if url == metadata_providers.PUBMED_EFETCH:
                    self.assertTrue(accept_xml)
                    return pubmed_xml
                if url.startswith(metadata_providers.CROSSREF_WORKS):
                    return {"message": {}}
                raise AssertionError(url)

            auditor = publication_audit.AsyncPublicationAuditor(
                object(),
                scite=scite,
                clock=lambda: datetime(2026, 9, 10, tzinfo=timezone.utc),
            )
            with mock.patch.object(
                publication_audit.metadata_providers,
                "fetch_http",
                side_effect=fetch,
            ):
                first = await auditor.audit(doi="10.1000/a", pmid="100")
                second = await auditor.audit(doi="10.1000/a", pmid="100")

            self.assertIs(first, second)
            self.assertEqual(len(calls), 2)
            self.assertEqual(scite.calls, [("10.1000/a",)])
            self.assertEqual(first.retraction.status, "reported_retracted")
            self.assertEqual(first.commentary.status, "critical_citation_found")
            self.assertEqual(first.checked_at, "2026-09-10T00:00:00Z")

        with self.subTest(case="title_lookup_payloads_are_reused_for_audit"):
            doi = "10.1000/reused"
            calls: list[str] = []
            scite = FakeScite({"total": 0, "contradicting": 0})
            pubmed_xml = f"""
        <PubmedArticleSet><PubmedArticle>
          <MedlineCitation>
            <PMID>100</PMID>
            <Article>
              <ArticleTitle>Reusable Provider Evidence</ArticleTitle>
              <Journal><JournalIssue><PubDate><Year>2026</Year></PubDate>
              </JournalIssue></Journal>
              <AuthorList><Author><LastName>Example</LastName></Author></AuthorList>
            </Article>
            <CommentsCorrectionsList>
              <CommentsCorrections RefType="RetractionIn"><PMID>200</PMID></CommentsCorrections>
            </CommentsCorrectionsList>
          </MedlineCitation>
          <PubmedData><ArticleIdList><ArticleId IdType="doi">{doi}</ArticleId>
          </ArticleIdList></PubmedData>
        </PubmedArticle></PubmedArticleSet>
        """
            crossref_message = {
                "DOI": doi,
                "title": ["Reusable Provider Evidence"],
                "issued": {"date-parts": [[2026]]},
                "container-title": ["Example Journal"],
                "author": [{"family": "Example"}],
            }

            async def fetch(_client, url, _params, accept_xml, *_args):
                calls.append(url)
                if url == metadata_providers.PUBMED_EFETCH:
                    self.assertTrue(accept_xml)
                    return pubmed_xml
                if url == metadata_providers.CROSSREF_WORKS_QUERY:
                    return {"message": {"items": [crossref_message]}}
                raise AssertionError(f"unexpected duplicate request: {url}")

            run_payloads = metadata_providers.RunProviderPayloads()
            pubmed = metadata_providers.PubMedAdapter(
                object(),
                object(),
                "",
                "",
                None,
                {},
                {},
                run_payloads,
            )
            crossref = metadata_providers.CrossrefAdapter(
                object(),
                object(),
                None,
                {},
                None,
                run_payloads,
            )
            auditor = publication_audit.AsyncPublicationAuditor(
                object(),
                run_payloads=run_payloads,
                scite=scite,
            )

            with (
                mock.patch.object(
                    metadata_providers.PubMedAdapter,
                    "ids_by_title",
                    mock.AsyncMock(return_value=["100"]),
                ),
                mock.patch.object(
                    metadata_providers,
                    "fetch_http",
                    side_effect=fetch,
                ),
            ):
                pubmed_lookup = await pubmed.records_by_title(
                    "Reusable Provider Evidence"
                )
                crossref_lookup = await crossref.records_by_title(
                    "Reusable Provider Evidence"
                )
                result = await auditor.audit(doi=doi)

            self.assertEqual(len(pubmed_lookup.records), 1)
            self.assertEqual(len(crossref_lookup.records), 1)
            self.assertEqual(calls.count(metadata_providers.PUBMED_EFETCH), 1)
            self.assertEqual(calls.count(metadata_providers.CROSSREF_WORKS_QUERY), 1)
            self.assertEqual(scite.calls, [(doi,)])
            self.assertEqual(result.retraction.status, "reported_retracted")


class AsyncSciteTalliesClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_bulk_requests_are_chunked_and_persistently_cached(self) -> None:
        requested: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.assertIsInstance(body, list)
            self.assertNotIn("authorization", request.headers)
            requested.append(body)
            return httpx.Response(
                200,
                json={
                    "tallies": {
                        doi: {"doi": doi, "total": 1, "contradicting": 0}
                        for doi in body
                    }
                },
            )

        dois = [f"10.1000/item-{index}" for index in range(501)]
        fixed = lambda: datetime(2026, 9, 14, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite"
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                cache = metadata_providers.MetadataCache(path)
                cache.open()
                try:
                    first = await publication_audit.AsyncSciteTalliesClient(
                        client,
                        cache=cache,
                        wall_clock=fixed,
                    ).fetch_many(dois)
                finally:
                    cache.close()

                cache = metadata_providers.MetadataCache(path)
                cache.open()
                try:
                    second = await publication_audit.AsyncSciteTalliesClient(
                        client,
                        cache=cache,
                        wall_clock=fixed,
                    ).fetch_many(reversed(dois))
                finally:
                    cache.close()

        self.assertEqual([len(batch) for batch in requested], [500, 1])
        self.assertEqual(len(first), 501)
        self.assertTrue(all(result.checked for result in second.values()))

    async def test_retry_after_is_honored_without_ten_second_truncation(self) -> None:
        calls = 0
        elapsed = 0.0
        sleeps: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "47"})
            doi = json.loads(request.content)[0]
            return httpx.Response(
                200,
                json={"tallies": {doi: {"doi": doi, "total": 0}}},
            )

        async def sleep(delay: float) -> None:
            nonlocal elapsed
            sleeps.append(delay)
            elapsed += delay

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            results = await publication_audit.AsyncSciteTalliesClient(
                client,
                sleep=sleep,
                monotonic=lambda: elapsed,
                jitter=lambda: 0.0,
            ).fetch_many(["10.1000/rate-limited"])

        self.assertEqual(sleeps, [47.0])
        self.assertEqual(calls, 2)
        self.assertTrue(results["10.1000/rate-limited"].checked)


class PublicationAuditReportTests(unittest.TestCase):
    def test_report_values_keep_formal_and_commentary_results_separate(self) -> None:
        audit = publication_audit.incomplete_audit(
            clock=lambda: datetime(2026, 9, 10, tzinfo=timezone.utc),
        )

        values = audit.report_values("published_")

        self.assertEqual(values["published_retraction_status"], "check_incomplete")
        self.assertEqual(values["published_commentary_status"], "check_incomplete")
        self.assertFalse(any("pubpeer" in name for name in values))
        self.assertEqual(
            tuple(values),
            publication_audit.report_field_names("published_"),
        )

    def test_compact_report_values_group_sparse_evidence_and_notes(self) -> None:
        detailed = publication_audit.empty_report_values()
        detailed.update(
            {
                "retraction_status": "reported_retracted",
                "retraction_evidence": "pubmed:retraction:PMID:200",
                "retraction_note": "formal notice found",
                "commentary_status": "critical_citation_found",
                "pubmed_comment_links": "CommentIn:PMID:300",
                "scite_total": 8,
                "scite_contradicting": 0,
                "commentary_note": "citation context found",
                "audit_checked_at": "2026-09-10T00:00:00Z",
            }
        )

        values = publication_audit.compact_report_values(detailed)

        self.assertEqual(
            tuple(values),
            publication_audit.compact_report_field_names(),
        )
        self.assertEqual(values["retraction_status"], "reported_retracted")
        self.assertEqual(
            values["commentary_status"],
            "critical_citation_found",
        )
        self.assertEqual(
            json.loads(str(values["audit_evidence"])),
            {
                "retraction": "pubmed:retraction:PMID:200",
                "pubmed_comments": "CommentIn:PMID:300",
                "scite": {"total": 8, "contradicting": 0},
            },
        )
        self.assertEqual(
            json.loads(str(values["audit_note"])),
            {
                "retraction": "formal notice found",
                "commentary": "citation context found",
            },
        )


# Retraction interpretation

class RetractionAuditTests(unittest.TestCase):
    def test_publication_version_links_are_not_corrections_or_retractions(self) -> None:
        for relation, kind in (("UpdateIn", "version_update_available"), ("UpdateOf", "updated_version")):
            with self.subTest(relation=relation):
                xml = f'''<PubmedArticleSet><PubmedArticle><MedlineCitation>
                <Article><PublicationTypeList><PublicationType>Preprint</PublicationType></PublicationTypeList></Article>
                <CommentsCorrectionsList><CommentsCorrections RefType="{relation}"><PMID>200</PMID></CommentsCorrections></CommentsCorrectionsList>
                </MedlineCitation></PubmedArticle></PubmedArticleSet>'''
                result = publication_status.assess_retractions(pubmed_xml=xml, pubmed_checked=True)
                self.assertEqual(result.status, "no_retraction_signal")
                self.assertIn(kind, result.evidence_text)
                self.assertIn("version", result.note.lower())
                incomplete = publication_status.assess_retractions(pubmed_xml=xml, pubmed_checked=True, crossref_expected=True)
                self.assertEqual(incomplete.status, "check_incomplete")

    def test_formal_update_direction(self) -> None:
        with self.subTest(case="crossref_updated_by_retraction_marks_target_retracted"):
            assessment = publication_status.assess_retractions(
                crossref_message={
                    "updated-by": [
                        {
                            "DOI": "10.1000/retraction",
                            "type": "retraction",
                            "source": "retraction-watch",
                            "updated": {"date-parts": [[2025, 2, 3]]},
                        }
                    ]
                },
                crossref_checked=True,
            )

            self.assertEqual(assessment.status, "reported_retracted")
            self.assertIn("10.1000/retraction", assessment.evidence_text)
            self.assertIn("source=retraction-watch", assessment.evidence_text)

        with self.subTest(case="crossref_update_to_marks_current_record_as_notice"):
            assessment = publication_status.assess_retractions(
                crossref_message={
                    "update-to": [
                        {
                            "DOI": "10.1000/original",
                            "type": "retraction",
                            "source": "publisher",
                        }
                    ]
                },
                crossref_checked=True,
            )

            self.assertEqual(assessment.status, "retraction_notice_document")

        with self.subTest(case="pubmed_retraction_and_republication_are_not_flattened"):
            xml = """
        <PubmedArticleSet><PubmedArticle><MedlineCitation>
          <PMID>100</PMID><Article><PublicationTypeList>
            <PublicationType>Retracted Publication</PublicationType>
          </PublicationTypeList></Article>
          <CommentsCorrectionsList>
            <CommentsCorrections RefType="RetractedandRepublishedIn">
              <PMID>200</PMID>
            </CommentsCorrections>
          </CommentsCorrectionsList>
        </MedlineCitation></PubmedArticle></PubmedArticleSet>
        """

            assessment = publication_status.assess_retractions(
                pubmed_xml=xml,
                pubmed_checked=True,
            )

            self.assertEqual(
                assessment.status,
                "retracted_then_reinstated_or_republished",
            )
            self.assertIn("PMID:200", assessment.evidence_text)


    def test_expression_of_concern_direction(self) -> None:
        with self.subTest(case="expression_of_concern_is_distinct_from_retraction"):
            xml = """
        <PubmedArticleSet><PubmedArticle><MedlineCitation>
          <PMID>100</PMID>
          <CommentsCorrectionsList>
            <CommentsCorrections RefType="ExpressionOfConcernIn">
              <PMID>201</PMID>
            </CommentsCorrections>
          </CommentsCorrectionsList>
        </MedlineCitation></PubmedArticle></PubmedArticleSet>
        """

            assessment = publication_status.assess_retractions(
                pubmed_xml=xml,
                pubmed_checked=True,
            )

            self.assertEqual(assessment.status, "expression_of_concern")

        with self.subTest(case="expression_of_concern_notice_is_not_target_article"):
            xml = """
        <PubmedArticleSet><PubmedArticle><MedlineCitation>
          <PMID>201</PMID><Article><PublicationTypeList>
            <PublicationType>Expression of Concern</PublicationType>
          </PublicationTypeList></Article>
          <CommentsCorrectionsList>
            <CommentsCorrections RefType="ExpressionOfConcernFor">
              <PMID>100</PMID>
            </CommentsCorrections>
          </CommentsCorrectionsList>
        </MedlineCitation></PubmedArticle></PubmedArticleSet>
        """

            assessment = publication_status.assess_retractions(
                pubmed_xml=xml,
                pubmed_checked=True,
            )

            self.assertEqual(
                assessment.status,
                "expression_of_concern_notice_document",
            )


    def test_negative_and_incomplete_audit_states(self) -> None:
        with self.subTest(case="checked_record_without_update_has_explicit_negative_signal"):
            assessment = publication_status.assess_retractions(
                crossref_message={},
                crossref_checked=True,
            )

            self.assertEqual(assessment.status, "no_retraction_signal")
            self.assertIn("not proof", assessment.note)

        with self.subTest(case="unavailable_sources_are_incomplete_not_negative"):
            assessment = publication_status.assess_retractions()

            self.assertEqual(assessment.status, "check_incomplete")

        with self.subTest(case="partial_source_failure_does_not_become_clean_audit"):
            assessment = publication_status.assess_retractions(
                crossref_message={},
                crossref_checked=True,
                crossref_expected=True,
                pubmed_expected=True,
                pubmed_checked=False,
            )

            self.assertEqual(assessment.status, "check_incomplete")


if __name__ == "__main__":
    unittest.main()

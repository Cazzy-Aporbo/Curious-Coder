# Literature snapshot and reuse boundary

This directory contains a bounded processed snapshot from the Europe PMC REST API. It is not a systematic review or a complete sample of the indexed literature.

- `corpus.json` contains 156 deduplicated records, bibliographic attribution, license metadata, editorial-notice links, and permitted abstract text.
- `queries.json` contains six authored known-item questions and explicit target identifiers. The targets were deliberately added to the corpus; they are not blinded or complete relevance judgments.
- `manifest.json` records acquisition time, exact broad queries, page limits, cursor trace, response hashes, corpus/query hashes, and the source audit.

Abstract text is retained only when the API license field is `cc by` or `cc0`. Records with other or missing license metadata remain title/metadata-only, even if a full text can be read freely online. Follow each record's source link and license statement for its applicable terms. Credit the authors and original publication when reusing included material; this repository does not claim ownership of their work.

Normalization removes presentation markup and collapses whitespace; the loader also removes escaped presentation tags before indexing. Scientific text is not rewritten or completed by a generative model. Raw API responses are hashed but not republished wholesale, and author affiliation/contact fields are not extracted.

The query window uses first-publication date, while the retained `year` field is the API's `pubYear`. Online-first and issue dates can differ. All records in this snapshot are provider-labeled English. Nine abstracts are absent upstream, 87 are withheld under the reuse rule, and 60 are included. Those states must remain distinguishable in downstream analysis.

The supplied manifest verifies the retained processed bytes. It is not an independent signature of the provider, a guarantee that upstream metadata never changes, or a certification of scientific validity. Reacquire into a new snapshot and compare provenance rather than silently replacing evidence.

Source documentation: [Europe PMC API](https://europepmc.org/restfulwebservice), [approved automated access routes](https://europepmc.org/developers), and [copyright policy](https://europepmc.org/copyright). The accompanying [retrieval study](../../studies/evidence_retrieval.md) explains the evaluation, latency measurements, and unresolved biases.

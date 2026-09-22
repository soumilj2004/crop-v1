# Recommendation Engine Sourcing Audit

Generated during handoff review, 2026-08-30. Covers `knowledge/knowledge_base.py`
(treatment plans) and `knowledge/engine.py` (weather risk thresholds).

## Summary

The knowledge base's only sourcing statement is the module docstring:

> "Source: Consolidated agronomy field-trial and extension guidance.
> All dosages and timings are from real extension recommendations."

No specific extension service, university, regulatory body, product label, or
publication is cited anywhere in either file. The dosage figures themselves
read as plausible, standard-formulation numbers (e.g. Tricyclazole 75% WP at
0.6 g/L for rice blast, Copper oxychloride 50% WP at 2.5-3 g/L for bacterial
blight, Propiconazole 25% EC at 1 mL/L for wheat leaf rust) that are
consistent with commonly published extension-service ranges -- but as
written, none of this is independently verifiable from the repo alone.

**53 chemical/dosage entries** were catalogued across Rice (5 diseases x 3
stages) and Wheat (4 diseases x 3 stages); the correctly-encoded "None / N/A"
entries (e.g. no antiviral for Tungro, no foliar rescue for Loose Smut or
Crown & Root Rot once systemic, seed-treatment-only guidance for next-season
prevention) are a genuine strength -- the engine does not overstate what
chemistry can do. That correctness doesn't substitute for a citation trail on
the entries that DO prescribe an active ingredient and rate.

## Weather risk thresholds (`engine.py`)

Five rule-based thresholds (fungal: humidity >=80%, 22-32C; bacterial:
humidity >=75%, rainfall >5mm; viral (vector-driven): 25-32C; seedborne
fungal: 18-24C, humidity >=60%; soilborne fungal: rainfall >15mm or humidity
>=85%). Same situation: plausible agronomic logic, zero citation.

## Recommendation

Before this ships or goes into the paper as validated guidance:

1. For each `chemical`/`dosage` pair, attach a source -- ideally a specific
   extension bulletin, ICAR/IRRI/CIMMYT guidance document, or peer-reviewed
   field trial, with a URL or DOI where available.
2. Where a real source can't be found or confirmed, mark the entry
   explicitly as "unverified / needs sourcing" rather than presenting it
   with the same confidence as a cited one.
3. Regional caveat: dosages and product availability vary by country/region;
   the current text doesn't specify which regulatory regime these rates
   assume (see handoff section on location handling -- do not assume the
   developer's location is the deployment region).
4. This entire knowledge base should be treated as **implemented rule-based
   logic**, not **agronomically validated prescriptions**, until step 1 is
   done -- consistent with the project's own "RULE-BASED != MACHINE
   LEARNING" / "PREDICTION != AGRONOMIC VALIDATION" distinctions.

No code changes were made as part of this audit -- it's a documentation
pass flagging what needs sourcing before the numbers are presented as
verified in the paper or a real deployment.

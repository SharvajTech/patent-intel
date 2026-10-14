"""
Patent Intel Agent - Synthetic sample source (Part 3)

IMPORTANT: These are NOT real patents. This module generates plausible-looking
but entirely fabricated patent records so the cleaning/chunking/embedding
pipeline can be exercised end-to-end without network access or an API key.

Use it for smoke tests and demos only. For real work, point build_dataset.py
at a real source (--source local / patentsview / hupd).

The generator deliberately injects dirty data (duplicate IDs, duplicate
content, empty abstracts, markup, stub records) so the cleaning stage has
something real to remove.
"""
import random
from typing import Any, Dict, List

_DOMAINS = [
    ("battery thermal management", "electric vehicle battery packs",
     "a coolant manifold", "thermal runaway propagation"),
    ("federated model training", "distributed edge devices",
     "a gradient aggregation server", "client data leakage"),
    ("microfluidic assay", "point-of-care diagnostic cartridges",
     "a capillary valve array", "sample cross-contamination"),
    ("beamforming antenna", "millimeter-wave base stations",
     "a phased array controller", "multipath interference"),
    ("additive manufacturing", "metal powder bed fusion systems",
     "a laser scan compensator", "residual stress warping"),
    ("crop yield prediction", "precision agriculture platforms",
     "a multispectral imaging unit", "sensor drift under cloud cover"),
    ("solid-state electrolyte", "lithium metal cells",
     "a ceramic separator layer", "dendrite formation"),
    ("speech enhancement", "far-field voice interfaces",
     "an adaptive noise estimator", "reverberation smearing"),
    ("surgical navigation", "image-guided orthopedic procedures",
     "a fiducial tracking module", "registration drift"),
    ("carbon capture", "flue gas treatment installations",
     "an amine regeneration loop", "solvent degradation"),
]

_ASSIGNEES = [
    "Northfield Systems Inc.", "Aequor Labs GmbH", "Kite Ridge Technologies",
    "Sendai Micro Devices K.K.", "Vantage Photonics Ltd.", "Orbital Materials Co.",
]


def _make_description(domain, rng: random.Random) -> str:
    topic, application, component, problem = domain
    paras = [
        f"FIELD OF THE INVENTION\n\nThe present disclosure relates generally to "
        f"{topic}, and more particularly to systems and methods employing "
        f"{component} in {application}.",

        f"BACKGROUND\n\nConventional approaches to {topic} suffer from "
        f"{problem}. Prior systems typically rely on fixed-parameter control "
        f"schemes that cannot adapt to changing operating conditions. As a "
        f"result, performance degrades over the service life of the apparatus, "
        f"and manufacturers are forced to over-specify components to maintain "
        f"acceptable margins. There remains a need for an approach that "
        f"mitigates {problem} without increasing system cost or complexity.",

        f"SUMMARY\n\nIn one aspect, an apparatus is provided comprising "
        f"{component} coupled to a controller configured to receive sensor "
        f"data indicative of an operating state and to adjust an actuation "
        f"parameter in response. In a further aspect, the controller applies "
        f"a predictive model trained on historical operating data to "
        f"anticipate {problem} before it manifests.",

        f"DETAILED DESCRIPTION\n\nReferring now to the embodiments, "
        f"{component} is disposed within a housing and thermally coupled to a "
        f"substrate. A sensor array samples at least one operating parameter "
        f"at a first sampling rate. The controller compares the sampled value "
        f"against a threshold derived from a calibration routine executed at "
        f"initialization. Where the sampled value exceeds the threshold, the "
        f"controller issues a compensation signal. In certain embodiments the "
        f"compensation signal modulates a duty cycle; in other embodiments it "
        f"selects among a plurality of predetermined operating profiles. The "
        f"skilled artisan will appreciate that the described arrangement may "
        f"be adapted to {application} of varying scale without departing from "
        f"the scope of the disclosure. Numerous modifications and variations "
        f"are possible in light of the above teachings, and the embodiments "
        f"were chosen to best explain the principles involved.",
    ]
    rng.shuffle(paras[3:])
    return "\n\n".join(paras)


def _make_claims(domain, rng: random.Random) -> str:
    topic, application, component, problem = domain
    n_claims = rng.randint(4, 9)
    claims = [
        f"1. An apparatus for {topic}, comprising: {component}; a sensor "
        f"configured to generate a signal indicative of an operating state; "
        f"and a controller operatively coupled to the sensor and configured "
        f"to adjust an actuation parameter based at least in part on the signal."
    ]
    templates = [
        "The apparatus of claim 1, wherein the controller comprises a "
        "predictive model trained on historical operating data.",
        "The apparatus of claim 1, further comprising a memory storing a "
        "plurality of predetermined operating profiles.",
        f"The apparatus of claim 1, wherein the actuation parameter is "
        f"selected to reduce {problem}.",
        "The apparatus of claim 1, wherein the sensor samples at a first "
        "sampling rate during a calibration mode and a second sampling rate "
        "during a steady-state mode.",
        f"A method of operating {component}, the method comprising: receiving "
        f"a sensor signal; comparing the sensor signal to a threshold; and "
        f"issuing a compensation signal when the threshold is exceeded.",
        "The method of claim 5, further comprising deriving the threshold "
        "from a calibration routine executed at initialization.",
        f"A non-transitory computer-readable medium storing instructions that, "
        f"when executed, cause a processor to control {component} in "
        f"{application}.",
        "The apparatus of claim 1, wherein the controller modulates a duty "
        "cycle of a drive signal.",
    ]
    rng.shuffle(templates)
    for i, body in enumerate(templates[: n_claims - 1], start=2):
        claims.append(f"{i}. {body}")
    return "\n\n".join(claims)


def generate_sample_records(count: int = 300, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate `count` synthetic raw patent records, plus deliberate dirty
    records so the cleaning stage has real work to do.
    """
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []

    for i in range(count):
        domain = _DOMAINS[i % len(_DOMAINS)]
        topic, application, component, problem = domain
        variant = i // len(_DOMAINS) + 1
        pid = f"US{10_000_000 + i * 37:,}".replace(",", "")
        title = (
            f"{topic.title()} system with {component} for {application} "
            f"(configuration {variant})"
        )
        abstract = (
            f"A system and method for {topic} are disclosed. The system "
            f"includes {component} coupled to a controller that adjusts an "
            f"actuation parameter responsive to sensed operating conditions, "
            f"thereby mitigating {problem} in {application}. Embodiments "
            f"provide improved stability across a range of duty cycles "
            f"without requiring additional hardware."
        )
        records.append({
            "patent_id": f"{pid} B2",
            "title": title,
            "abstract": abstract,
            "description": _make_description(domain, rng),
            "claims": _make_claims(domain, rng),
            "date": f"20{rng.randint(15, 24):02d}-{rng.randint(1,12):02d}-"
                    f"{rng.randint(1,28):02d}",
            "assignee": rng.choice(_ASSIGNEES),
        })

    # --- deliberate dirty data, to exercise the cleaning stage -------------
    if records:
        # 1. exact duplicate ID, sparser content (should lose to the original)
        dup = dict(records[0])
        dup["description"] = ""
        dup["claims"] = ""
        records.append(dup)

        # 2. same title+abstract under a different ID (content duplicate)
        content_dup = dict(records[1])
        content_dup["patent_id"] = "US 9,999,111 A1"
        records.append(content_dup)

        # 3. record with markup that must be stripped
        marked = dict(records[2])
        marked["patent_id"] = "US 9,999,222 A1"
        marked["abstract"] = (
            "<p>A <b>system</b> for thermal control.</p><claim-text>The "
            "apparatus  includes&nbsp;a  manifold (FIG. 3) and a "
            "controller.</claim-text>"
        )
        records.append(marked)

    # 4. structurally broken / unusable records
    records.extend([
        {"patent_id": "", "title": "No identifier", "abstract": "x " * 40},
        {"patent_id": "US 9,999,333 A1", "title": "Ab", "abstract": "y " * 40},
        {"patent_id": "US 9,999,444 A1", "title": "Valid title here",
         "abstract": "too short", "description": "", "claims": ""},
        {"patent_id": "US 9,999,555 A1", "title": "Empty everything",
         "abstract": "", "description": "", "claims": ""},
        "not-a-dict-at-all",
    ])

    return records

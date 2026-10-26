"""A small engineering-documentation knowledge base (original summaries written
for this project, covering power-electronics and power-infrastructure topics),
used as the retrieval corpus for the RAG pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


DOCUMENTS: list[Document] = [
    Document(
        "sst-overview",
        "Solid-State Transformer Overview",
        "A solid-state transformer replaces the bulky iron core and copper windings of a "
        "conventional transformer with power-electronic converters and a high-frequency "
        "transformer stage. This lets it change voltage levels, convert between AC and DC, "
        "and regulate power flow in a single compact unit, at a fraction of the size and "
        "weight of a traditional line-frequency transformer rated for the same power.",
    ),
    Document(
        "sic-semiconductors",
        "Silicon Carbide Power Semiconductors",
        "Silicon carbide, or SiC, is a wide-bandgap semiconductor material used to build power "
        "transistors and diodes that switch faster and tolerate higher voltages and "
        "temperatures than traditional silicon devices. SiC devices reduce switching losses "
        "in power converters, which allows a converter to run at a higher switching frequency "
        "and therefore use smaller magnetic and capacitive components.",
    ),
    Document(
        "gan-semiconductors",
        "Gallium Nitride Power Semiconductors",
        "Gallium nitride, or GaN, is another wide-bandgap semiconductor, generally used at "
        "lower voltages than silicon carbide but with even faster switching speed. GaN "
        "transistors are common in compact chargers and lower-power converters where minimizing "
        "size and switching loss matters more than handling very high voltage.",
    ),
    Document(
        "mv-power-conversion",
        "Medium Voltage Power Conversion",
        "Medium voltage power conversion typically covers equipment operating between roughly "
        "1 kV and 35 kV, a range common in data center campus distribution, industrial plants, "
        "and utility feeders. Converters in this range must manage higher insulation "
        "requirements, more complex protection schemes, and often use multilevel or modular "
        "converter topologies to share voltage stress across multiple semiconductor devices.",
    ),
    Document(
        "ev-fast-charging",
        "EV DC Fast Charging Architecture",
        "A DC fast charging station converts AC grid power to DC at the voltage and current an "
        "electric vehicle's battery requires, allowing the vehicle to charge directly without "
        "relying on its onboard charger. High power DC fast chargers increasingly use solid "
        "state power conversion stages to reach higher power levels in a smaller footprint "
        "while meeting efficiency and thermal targets for outdoor, high duty-cycle operation.",
    ),
    Document(
        "data-center-power",
        "Data Center Power Distribution",
        "Modern data centers distribute power from the utility grid through several conversion "
        "and switching stages before it reaches server racks, typically stepping down voltage, "
        "converting AC to DC, and conditioning power quality along the way. As AI training "
        "clusters push rack power density far higher than traditional data center loads, the "
        "power infrastructure feeding those racks must support faster deployment and much "
        "higher power density than legacy designs.",
    ),
    Document(
        "battery-storage",
        "Battery Energy Storage Systems",
        "A battery energy storage system, or BESS, uses a power conversion system to move "
        "energy between a battery and an AC grid connection, charging the battery when supply "
        "is abundant and discharging it to support the grid or a facility during peak demand. "
        "The power conversion system's efficiency and response speed directly determine how "
        "effectively the stored energy can be used for grid services like frequency regulation.",
    ),
    Document(
        "grid-interconnection",
        "Grid Interconnection Standards",
        "Equipment connecting to the electric grid, including inverters, converters, and solid "
        "state transformers, must meet interconnection standards covering voltage and frequency "
        "ride-through, harmonic distortion limits, and protective relaying behavior during grid "
        "faults. Meeting these standards is what allows new power infrastructure to be certified "
        "and deployed on a utility or campus electrical system.",
    ),
    Document(
        "converter-topologies",
        "Power Converter Topologies",
        "Common power converter topologies include the buck converter and boost converter for "
        "simple DC-DC step-down or step-up conversion, and the dual active bridge for "
        "bidirectional, galvanically isolated DC-DC conversion between two DC buses. The dual "
        "active bridge topology is widely used inside solid-state transformers because it can "
        "transfer power in either direction through a high-frequency transformer stage.",
    ),
    Document(
        "thermal-management",
        "Thermal Management in Power Electronics",
        "Every switching event in a power semiconductor produces heat, and power electronics "
        "designs must remove that heat fast enough to keep junction temperatures within the "
        "device's rated limits. Thermal management approaches range from passive heatsinks and "
        "forced air cooling to liquid cold plates for high power density designs, with the "
        "choice driven by how much power loss must be removed per unit volume.",
    ),
    Document(
        "power-factor-correction",
        "Power Factor Correction",
        "Power factor correction circuitry shapes the input current drawn by a power converter "
        "so that it stays in phase with the input voltage, improving power factor and reducing "
        "harmonic distortion fed back onto the grid. Many grid-connected converters include an "
        "active power factor correction stage to meet utility and regulatory harmonic limits.",
    ),
    Document(
        "switching-losses",
        "Switching Frequency and Losses",
        "Raising a power converter's switching frequency shrinks the size of its magnetic "
        "components and capacitors, but it also increases switching losses in the "
        "semiconductors, since some energy is lost every time a device turns on or off. Wide "
        "bandgap devices such as SiC and GaN push this tradeoff further than silicon by cutting "
        "per-switching-event losses, enabling higher frequency operation at acceptable "
        "efficiency.",
    ),
    Document(
        "galvanic-isolation",
        "Galvanic Isolation",
        "Galvanic isolation electrically separates two parts of a circuit so that no direct "
        "conduction path exists between them, typically using a transformer or an optical or "
        "capacitive coupling element to transfer signal or power across the isolation boundary. "
        "Isolation is required between high voltage and low voltage sections of a power "
        "converter for both safety and to prevent ground loops.",
    ),
    Document(
        "transformer-efficiency",
        "Transformer Efficiency and Core Losses",
        "A transformer loses a small fraction of the power passing through it as heat, split "
        "between winding losses from resistive heating in the conductors and core losses from "
        "magnetizing the transformer's magnetic core at the operating frequency. Core losses "
        "generally rise with both frequency and magnetic flux density, which is one reason "
        "high-frequency transformer stages are designed with lower flux density cores.",
    ),
    Document(
        "solar-inverters",
        "Renewable Energy Integration and Solar Inverters",
        "A solar inverter converts the variable DC output of a photovoltaic array into grid "
        "compliant AC power, continuously tracking the array's maximum power point as sunlight "
        "and temperature change throughout the day. Large solar installations often use central "
        "or string inverters feeding into medium voltage transformers before connecting to the "
        "wider grid.",
    ),
    Document(
        "fault-protection",
        "Protection and Fault Handling in Power Systems",
        "Power electronic equipment must detect and respond to fault conditions such as "
        "short circuits, overvoltage, and overcurrent within microseconds to protect both "
        "itself and the connected grid or load. Protection schemes combine fast hardware "
        "level current limiting with higher level control logic that can trip a breaker or "
        "shut down a converter stage when a fault is confirmed.",
    ),
    Document(
        "modular-multilevel",
        "Modular Multilevel Converters",
        "A modular multilevel converter builds a high voltage waveform out of many small, "
        "identical submodules connected in series, each switching a small voltage step so that "
        "no single semiconductor device has to block the full system voltage. This modular "
        "approach scales well to very high voltage applications and allows a design to be "
        "built from the same submodule across many different system voltage ratings.",
    ),
    Document(
        "converter-controls",
        "Control Systems for Power Converters",
        "Power converter control systems regulate quantities like output voltage, current, or "
        "power by continuously measuring the converter's state and adjusting switching "
        "commands many thousands of times per second. Digital control, typically running on "
        "a microcontroller or DSP, has largely replaced analog control loops because it allows "
        "more complex control algorithms and easier integration with monitoring and protection.",
    ),
    Document(
        "emi-emc",
        "EMI and EMC in Power Electronics",
        "Fast switching in power converters generates electromagnetic interference, or EMI, "
        "which must be controlled through filtering and careful layout so that the equipment "
        "does not disrupt nearby electronics and meets electromagnetic compatibility, or EMC, "
        "regulatory limits. EMI filters are typically placed at a converter's input and output "
        "to attenuate high frequency noise before it reaches the surrounding system.",
    ),
    Document(
        "reliability-redundancy",
        "Reliability and Redundancy in Critical Power Infrastructure",
        "Power infrastructure supporting critical loads such as data centers is often designed "
        "with redundant conversion paths, so that a single component failure does not interrupt "
        "power to the load. Reliability engineering for this equipment focuses on predicting "
        "failure modes, reducing single points of failure, and enabling fast field "
        "replacement of failed modules without taking the whole system offline.",
    ),
]

DOCUMENTS_BY_ID = {doc.doc_id: doc for doc in DOCUMENTS}

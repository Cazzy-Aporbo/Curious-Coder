"""
WOMEN'S HEALTH GUT-BRAIN AXIS DIGITAL TWIN MODEL
Integrated Framework for Hormonal, Microbial, and Neural Interactions

This model incorporates sex-specific factors affecting the gut-brain axis:
- Estrobolome and hormone metabolism (Baker et al., 2017, Nat Rev Endocrinol)
- Menstrual cycle effects on gut permeability (Heitkemper et al., 2003, Aliment Pharmacol Ther)
- Sex hormones and microbiome interactions (Flores et al., 2012, PLoS One)
- HPA/HPG axis crosstalk (Oyola & Handa, 2017, Front Neuroendocrinol)
- Progesterone metabolites and GABA signaling (Belelli & Lambert, 2005, Nat Rev Neurosci)

Primary Literature References:
- Cryan & Dinan, 2012, Nature Reviews Neuroscience
- Mayer et al., 2014, Journal of Neuroscience
- Dinan & Cryan, 2017, Journal of Physiology
- Santos et al., 2019, Nature Reviews Gastroenterology & Hepatology
- Margolis et al., 2021, Cell

Author: Cazandra Aporbo
Version: 2.0.0
Python Requirements: 3.9+
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum, auto
import scipy.integrate as integrate
from scipy.stats import pearsonr, spearmanr, zscore
from scipy.signal import find_peaks, welch
from scipy.optimize import minimize, curve_fit
import networkx as nx
from collections import defaultdict, deque
import logging
from datetime import datetime, timedelta
import json
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiologicalReferences:
    """
    Validated reference ranges from peer-reviewed literature
    All values include citations to primary sources
    """
    
    # Sex hormone reference ranges (Mayo Clinic Laboratories, 2023)
    ESTRADIOL_FOLLICULAR = (30, 120)  # pg/mL (Stricker et al., 2006, Clin Chem Lab Med)
    ESTRADIOL_OVULATORY = (130, 370)  # pg/mL
    ESTRADIOL_LUTEAL = (70, 250)  # pg/mL
    ESTRADIOL_POSTMENOPAUSE = (0, 30)  # pg/mL
    
    PROGESTERONE_FOLLICULAR = (0.1, 0.9)  # ng/mL (Mesen & Young, 2015, Obstet Gynecol Clin North Am)
    PROGESTERONE_LUTEAL = (5, 25)  # ng/mL
    PROGESTERONE_POSTMENOPAUSE = (0.1, 0.5)  # ng/mL
    
    TESTOSTERONE_FEMALE = (15, 70)  # ng/dL (Davis et al., 2016, Lancet Diabetes Endocrinol)
    
    # Neurotransmitter CSF levels (Eisenhofer et al., 2004, Pharmacol Rev)
    SEROTONIN_CSF = (0.5, 2.0)  # nM
    DOPAMINE_CSF = (0.1, 1.0)  # nM
    GABA_CSF = (40, 150)  # nM
    GLUTAMATE_CSF = (5000, 15000)  # nM
    
    # Gut-derived metabolites (Schroeder & Bäckhed, 2016, Cell)
    SCFA_ACETATE = (40000, 80000)  # μM in colon
    SCFA_PROPIONATE = (10000, 30000)  # μM
    SCFA_BUTYRATE = (10000, 30000)  # μM
    
    # Inflammatory markers (Ridker, 2003, Circulation)
    CRP_NORMAL = (0, 3)  # mg/L
    IL6_NORMAL = (0, 7)  # pg/mL
    TNF_ALPHA_NORMAL = (0, 8.1)  # pg/mL
    
    # Stress hormones (Burke et al., 2005, Biol Psychiatry)
    CORTISOL_MORNING = (10, 20)  # μg/dL
    CORTISOL_EVENING = (2, 10)  # μg/dL

class MenstrualPhase(Enum):
    """
    Menstrual cycle phases with hormonal characteristics
    Based on Reed & Carr, 2018, StatPearls
    """
    MENSTRUAL = auto()  # Days 1-5
    FOLLICULAR = auto()  # Days 1-13
    OVULATORY = auto()  # Days 13-15
    LUTEAL = auto()  # Days 15-28
    ANOVULATORY = auto()  # No ovulation
    POSTMENOPAUSE = auto()
    PREGNANCY = auto()
    
    @property
    def gut_permeability_modifier(self) -> float:
        """
        Intestinal permeability changes across cycle
        Zhou & Verne, 2011, Pain Research and Treatment
        """
        modifiers = {
            MenstrualPhase.MENSTRUAL: 1.3,  # Increased permeability
            MenstrualPhase.FOLLICULAR: 0.9,
            MenstrualPhase.OVULATORY: 0.85,
            MenstrualPhase.LUTEAL: 1.1,
            MenstrualPhase.ANOVULATORY: 1.0,
            MenstrualPhase.POSTMENOPAUSE: 1.2,
            MenstrualPhase.PREGNANCY: 0.95
        }
        return modifiers[self]
    
    @property
    def microbiome_diversity_effect(self) -> float:
        """
        Microbiome alpha diversity changes
        García-Peñarrubia et al., 2022, Int J Mol Sci
        """
        effects = {
            MenstrualPhase.MENSTRUAL: 0.92,
            MenstrualPhase.FOLLICULAR: 1.05,
            MenstrualPhase.OVULATORY: 1.08,
            MenstrualPhase.LUTEAL: 0.95,
            MenstrualPhase.ANOVULATORY: 0.90,
            MenstrualPhase.POSTMENOPAUSE: 0.85,
            MenstrualPhase.PREGNANCY: 1.10
        }
        return effects[self]

@dataclass
class WomensHealthProfile:
    """
    Comprehensive women's health parameters for digital twin
    """
    
    # Basic demographics
    age: int = 30
    bmi: float = 23.0
    
    # Reproductive status
    menstrual_phase: MenstrualPhase = MenstrualPhase.FOLLICULAR
    cycle_day: int = 7
    cycle_length: int = 28
    cycle_regularity: float = 0.9  # 0-1, coefficient of variation
    
    # Hormonal measurements (with defaults in normal ranges)
    estradiol: float = 80.0  # pg/mL
    progesterone: float = 0.5  # ng/mL
    testosterone: float = 35.0  # ng/dL
    lh: float = 5.0  # mIU/mL
    fsh: float = 6.0  # mIU/mL
    prolactin: float = 15.0  # ng/mL
    cortisol: float = 12.0  # μg/dL
    
    # Thyroid function (Hollowell et al., 2002, J Clin Endocrinol Metab)
    tsh: float = 1.5  # mIU/L (0.4-4.0)
    free_t4: float = 1.2  # ng/dL (0.9-1.7)
    
    # Metabolic markers
    insulin: float = 8.0  # μIU/mL
    glucose: float = 85.0  # mg/dL
    hba1c: float = 5.2  # %
    
    # Reproductive history
    pregnancies: int = 0
    births: int = 0
    breastfeeding: bool = False
    contraception_type: str = "none"  # none, ocp, iud, other
    
    # Conditions
    pcos: bool = False  # Polycystic ovary syndrome
    endometriosis: bool = False
    pmdd: bool = False  # Premenstrual dysphoric disorder
    
    def calculate_allostatic_load(self) -> float:
        """
        Calculate allostatic load index for women
        Juster et al., 2010, Physiol Behav
        """
        load_score = 0.0
        
        # Neuroendocrine markers
        if self.cortisol > 20: load_score += 1
        if self.cortisol < 5: load_score += 1
        
        # Metabolic markers
        if self.bmi > 30: load_score += 1
        if self.glucose > 100: load_score += 1
        if self.hba1c > 5.7: load_score += 1
        
        # Cardiovascular (would need BP data)
        # Inflammatory (would need CRP, IL-6)
        
        return load_score / 5  # Normalize to 0-1

@dataclass
class SymptomTracking:
    """
    Daily symptom tracking for pattern recognition
    Validated scales from women's health research
    """
    
    # Gastrointestinal (Rome IV criteria, Lacy et al., 2016, Gastroenterology)
    bloating: float = 0.0  # 0-10 VAS
    abdominal_pain: float = 0.0
    constipation: float = 0.0
    diarrhea: float = 0.0
    nausea: float = 0.0
    
    # Mood/Psychological (PROMIS scales, Cella et al., 2010, Med Care)
    anxiety: float = 0.0  # 0-10
    depression: float = 0.0
    irritability: float = 0.0
    mood_swings: float = 0.0
    brain_fog: float = 0.0
    fatigue: float = 0.0
    
    # Women-specific (Harlow et al., 2017, J Women's Health)
    breast_tenderness: float = 0.0
    pelvic_pain: float = 0.0
    hot_flashes: float = 0.0
    night_sweats: float = 0.0
    headache: float = 0.0
    water_retention: float = 0.0
    
    # Sleep (Pittsburgh Sleep Quality Index, Buysse et al., 1989, Psychiatry Res)
    sleep_quality: float = 5.0  # 0-10
    sleep_hours: float = 7.0
    sleep_disruptions: int = 1
    
    # Stress (Perceived Stress Scale, Cohen et al., 1983, J Health Soc Behav)
    stress_level: float = 5.0  # 0-10
    
    # Sexual health (Female Sexual Function Index, Rosen et al., 2000, J Sex Marital Ther)
    libido: float = 5.0  # 0-10
    sexual_satisfaction: float = 5.0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_pms_severity(self) -> float:
        """
        Calculate PMS severity score
        Steiner et al., 2011, Int J Women's Health
        """
        physical = (self.bloating + self.breast_tenderness + 
                   self.headache + self.water_retention) / 4
        psychological = (self.anxiety + self.irritability + 
                        self.mood_swings + self.depression) / 4
        
        return (physical * 0.4 + psychological * 0.6)

@dataclass
class EstrobolomeProfile:
    """
    Estrobolome: gut bacteria that metabolize estrogen
    Plottel & Blaser, 2011, J Steroid Biochem Mol Biol
    Kwa et al., 2016, Maturitas
    """
    
    # Beta-glucuronidase producing bacteria (Ervin et al., 2019, PLoS Comput Biol)
    clostridia_abundance: float = 30.0  # % of total
    escherichia_abundance: float = 0.1  # %
    bacteroides_abundance: float = 25.0  # %
    bifidobacterium_abundance: float = 3.0  # %
    lactobacillus_abundance: float = 0.5  # %
    
    # Enzyme activity
    beta_glucuronidase_activity: float = 1.0  # Normalized
    beta_glucosidase_activity: float = 1.0
    
    # Estrogen metabolites (Fuhrman et al., 2014, J Clin Endocrinol Metab)
    estrone_gut: float = 50.0  # pg/mL
    estradiol_gut: float = 20.0  # pg/mL
    estriol_gut: float = 10.0  # pg/mL
    
    # Metabolite ratios
    two_oh_e1_16_oh_e1_ratio: float = 2.0  # Protective if >2
    
    def calculate_estrogen_recirculation(self, fiber_intake: float) -> float:
        """
        Calculate estrogen recirculation rate
        Gaskins et al., 2009, Am J Clin Nutr
        """
        # Base recirculation from beta-glucuronidase activity
        base_recirc = self.beta_glucuronidase_activity * 0.4
        
        # Fiber reduces recirculation (Rose et al., 1991, Am J Clin Nutr)
        fiber_effect = 1.0 - min(0.3, fiber_intake / 100)
        
        # Calculate total recirculation (0-1)
        return min(1.0, base_recirc * fiber_effect)
    
    def predict_hormone_sensitive_cancer_risk(self) -> float:
        """
        Estimate relative risk based on estrobolome
        Goedert et al., 2015, J Natl Cancer Inst
        """
        risk_score = 1.0
        
        # High beta-glucuronidase activity increases risk
        if self.beta_glucuronidase_activity > 1.5:
            risk_score *= 1.3
        
        # Protective metabolite ratio
        if self.two_oh_e1_16_oh_e1_ratio < 2.0:
            risk_score *= 1.2
        
        # Low Lactobacillus increases risk (Goedert et al., 2015)
        if self.lactobacillus_abundance < 0.1:
            risk_score *= 1.15
        
        return risk_score

class WomensGutBrainAxis:
    """
    Sex-specific gut-brain axis model incorporating hormonal influences
    """
    
    def __init__(self):
        self.profile = WomensHealthProfile()
        self.symptoms = SymptomTracking()
        self.estrobolome = EstrobolomeProfile()
        
        # Initialize subsystems
        self.hpa_axis = HPAAxis()
        self.hpg_axis = HPGAxis()
        self.vagus_nerve = VagusNerveSystem()
        self.enteric_ns = EntericNervousSystem()
        self.immune_system = NeuroImmuneSystem()
        
        # Microbiome components
        self.microbiome = MicrobiomeComposition()
        self.metabolome = MicrobialMetabolome()
        
        # Neural networks
        self.brain_regions = self._initialize_brain_regions()
        
        # Data storage for learning
        self.symptom_history = deque(maxlen=90)  # 3 months
        self.intervention_history = deque(maxlen=30)
        
    def _initialize_brain_regions(self) -> Dict[str, 'BrainRegion']:
        """
        Initialize brain regions with sex-specific characteristics
        McEwen & Milner, 2017, J Neurosci Res
        """
        regions = {}
        
        regions['prefrontal_cortex'] = BrainRegion(
            name='Prefrontal Cortex',
            estrogen_receptor_alpha=True,  # Shansky et al., 2004, Mol Psychiatry
            estrogen_receptor_beta=True,
            progesterone_receptor=True,
            androgen_receptor=True
        )
        
        regions['hippocampus'] = BrainRegion(
            name='Hippocampus',
            estrogen_receptor_alpha=True,  # High density
            estrogen_receptor_beta=True,
            progesterone_receptor=True,
            androgen_receptor=False,
            bdnf_expression=1.0  # Normalized, affected by estrogen
        )
        
        regions['amygdala'] = BrainRegion(
            name='Amygdala',
            estrogen_receptor_alpha=True,
            estrogen_receptor_beta=True,
            progesterone_receptor=True,  # High density
            androgen_receptor=True
        )
        
        regions['hypothalamus'] = BrainRegion(
            name='Hypothalamus',
            estrogen_receptor_alpha=True,  # Critical for feedback
            estrogen_receptor_beta=True,
            progesterone_receptor=True,
            androgen_receptor=True,
            gnrh_neurons=1000  # GnRH neurons for HPG axis
        )
        
        return regions
    
    def simulate_cycle_day(self, day: int) -> Dict[str, Any]:
        """
        Simulate one day of the menstrual cycle
        Returns predicted symptoms and biomarkers
        """
        
        # Update cycle phase
        self.profile.cycle_day = day
        self.profile.menstrual_phase = self._determine_phase(day)
        
        # Calculate hormonal levels (simplified sinusoidal model)
        # Based on Stricker et al., 2006, Clin Chem Lab Med
        if self.profile.menstrual_phase == MenstrualPhase.FOLLICULAR:
            self.profile.estradiol = 50 + 30 * np.sin(np.pi * day / 14)
            self.profile.progesterone = 0.5
        elif self.profile.menstrual_phase == MenstrualPhase.OVULATORY:
            self.profile.estradiol = 200 + 100 * np.sin(np.pi * (day - 12) / 3)
            self.profile.lh = 25 + 15 * np.sin(np.pi * (day - 13) / 2)
            self.profile.progesterone = 0.8
        elif self.profile.menstrual_phase == MenstrualPhase.LUTEAL:
            self.profile.estradiol = 120 - 40 * np.sin(np.pi * (day - 14) / 14)
            self.profile.progesterone = 2 + 10 * np.sin(np.pi * (day - 14) / 14)
        else:  # Menstrual
            self.profile.estradiol = 40
            self.profile.progesterone = 0.3
        
        # Calculate downstream effects
        results = {}
        
        # 1. Gut permeability modulation
        # Braniste et al., 2014, Sci Transl Med - estrogen protects barrier
        gut_permeability = self._calculate_gut_permeability()
        results['gut_permeability'] = gut_permeability
        
        # 2. Microbiome changes
        # Flores et al., 2012, PLoS One - estrogen affects diversity
        microbiome_state = self._update_microbiome_state()
        results['microbiome_diversity'] = microbiome_state['diversity']
        results['dysbiosis_index'] = microbiome_state['dysbiosis']
        
        # 3. Neurotransmitter synthesis
        # Barth et al., 2015, Front Neuroendocrinol - estrogen affects serotonin
        neurotransmitters = self._calculate_neurotransmitters()
        results['serotonin'] = neurotransmitters['serotonin']
        results['gaba'] = neurotransmitters['gaba']
        results['dopamine'] = neurotransmitters['dopamine']
        
        # 4. Inflammation status
        # Villa et al., 2015, Front Immunol - sex differences in immunity
        inflammation = self._calculate_inflammation()
        results['neuroinflammation'] = inflammation['neuroinflammation']
        results['cytokines'] = inflammation['cytokines']
        
        # 5. Stress response
        # Oyola & Handa, 2017, Front Neuroendocrinol - HPA/HPG interaction
        stress_response = self._calculate_stress_response()
        results['cortisol'] = stress_response['cortisol']
        results['cortisol_awakening_response'] = stress_response['car']
        
        # 6. Symptom predictions
        symptoms = self._predict_symptoms(results)
        results['predicted_symptoms'] = symptoms
        
        # 7. Metabolomics
        # Tejesvi et al., 2022, Front Microbiol - cycle affects metabolome
        metabolites = self._calculate_metabolome()
        results['scfa'] = metabolites['scfa']
        results['tryptophan_metabolites'] = metabolites['tryptophan']
        
        return results
    
    def _determine_phase(self, day: int) -> MenstrualPhase:
        """
        Determine menstrual phase from cycle day
        Bull et al., 2019, NPJ Digit Med
        """
        if self.profile.cycle_length < 21 or self.profile.cycle_length > 35:
            # Irregular cycle
            return MenstrualPhase.ANOVULATORY
        
        if day <= 5:
            return MenstrualPhase.MENSTRUAL
        elif day <= 13:
            return MenstrualPhase.FOLLICULAR
        elif day <= 15:
            return MenstrualPhase.OVULATORY
        else:
            return MenstrualPhase.LUTEAL
    
    def _calculate_gut_permeability(self) -> float:
        """
        Calculate intestinal permeability with hormonal modulation
        Zhou et al., 2010, Life Sci
        """
        # Base permeability
        base = 1.0
        
        # Estrogen protective effect (Collins et al., 2021, Trends Endocrinol Metab)
        estrogen_effect = 1.0 - (self.profile.estradiol / 200) * 0.2
        
        # Progesterone effect (less protective)
        prog_effect = 1.0 + (self.profile.progesterone / 20) * 0.1
        
        # Phase-specific modifier
        phase_modifier = self.profile.menstrual_phase.gut_permeability_modifier
        
        # Stress increases permeability (Vanuytsel et al., 2014, Gut)
        stress_effect = 1.0 + (self.profile.cortisol / 30) * 0.2
        
        # Microbiome effect (Ghosh et al., 2020, Cell Host Microbe)
        dysbiosis_effect = 1.0 + self.microbiome.dysbiosis_index * 0.3
        
        permeability = base * estrogen_effect * prog_effect * phase_modifier * stress_effect * dysbiosis_effect
        
        return max(0.5, min(2.0, permeability))
    
    def _update_microbiome_state(self) -> Dict[str, float]:
        """
        Update microbiome based on hormonal state
        García-Peñarrubia et al., 2022, Int J Mol Sci
        """
        state = {}
        
        # Estrogen increases diversity (Flores et al., 2012, PLoS One)
        diversity_base = 3.5  # Shannon index
        estrogen_diversity = diversity_base * (1 + self.profile.estradiol / 500)
        phase_effect = self.profile.menstrual_phase.microbiome_diversity_effect
        
        state['diversity'] = diversity_base * phase_effect
        
        # Calculate dysbiosis (lower diversity = higher dysbiosis)
        state['dysbiosis'] = max(0, (4 - state['diversity']) / 4)
        
        # Update specific taxa
        # Lactobacillus affected by estrogen (Muhleisen & Herbst-Kralovetz, 2016, Front Cell Infect Microbiol)
        self.microbiome.lactobacillus = 8.0 + np.log10(1 + self.profile.estradiol / 100)
        
        # Prevotella associated with PCOS (Torres et al., 2018, J Clin Endocrinol Metab)
        if self.profile.pcos:
            self.microbiome.prevotella = 8.5
        else:
            self.microbiome.prevotella = 7.0
        
        return state
    
    def _calculate_neurotransmitters(self) -> Dict[str, float]:
        """
        Calculate neurotransmitter levels with sex hormone modulation
        """
        nt = {}
        
        # Serotonin synthesis affected by estrogen
        # Bethea et al., 2002, Biol Psychiatry - estrogen increases TPH2
        tph2_expression = 1.0 + (self.profile.estradiol / 200) * 0.3
        nt['serotonin'] = 1.5 * tph2_expression
        
        # GABA affected by progesterone metabolites
        # Bäckström et al., 2014, Psychoneuroendocrinology - allopregnanolone
        allopregnanolone = self.profile.progesterone * 0.1  # Simplified conversion
        gaba_potentiation = 1.0 + allopregnanolone * 0.2
        nt['gaba'] = 100 * gaba_potentiation
        
        # Dopamine modulated by estrogen
        # Jacobs & D'Esposito, 2011, J Neurosci - estrogen affects DA
        nt['dopamine'] = 0.5 * (1 + self.profile.estradiol / 300)
        
        # Glutamate (excitatory)
        nt['glutamate'] = 10000 * (1 + self.symptoms.anxiety / 20)
        
        return nt
    
    def _calculate_inflammation(self) -> Dict[str, Any]:
        """
        Calculate inflammatory status with sex differences
        Klein & Flanagan, 2016, Nat Rev Immunol
        """
        inflammation = {}
        
        # Baseline cytokines
        cytokines = {
            'il6': 2.0,
            'tnf_alpha': 5.0,
            'il1_beta': 3.0,
            'il10': 10.0  # Anti-inflammatory
        }
        
        # Estrogen generally anti-inflammatory (Villa et al., 2015, Front Immunol)
        estrogen_factor = 1.0 - (self.profile.estradiol / 400) * 0.2
        
        # Progesterone can be pro-inflammatory in some contexts
        prog_factor = 1.0 + (self.profile.progesterone / 30) * 0.1
        
        # Microbiome-derived inflammation (LPS)
        # Cani et al., 2007, Diabetes - metabolic endotoxemia
        lps_inflammation = self.microbiome.gram_negative_abundance * 0.01
        
        # Apply modulations
        for cytokine in ['il6', 'tnf_alpha', 'il1_beta']:
            cytokines[cytokine] *= estrogen_factor * prog_factor * (1 + lps_inflammation)
        
        inflammation['cytokines'] = cytokines
        
        # Calculate neuroinflammation score
        pro_inflammatory = (cytokines['il6'] + cytokines['tnf_alpha'] + 
                           cytokines['il1_beta']) / 3
        anti_inflammatory = cytokines['il10']
        
        inflammation['neuroinflammation'] = pro_inflammatory / (anti_inflammatory + 1)
        
        return inflammation
    
    def _calculate_stress_response(self) -> Dict[str, float]:
        """
        Calculate HPA axis response with HPG interaction
        Oyola & Handa, 2017, Front Neuroendocrinol
        """
        stress = {}
        
        # Circadian cortisol rhythm (Kudielka et al., 2009, Psychoneuroendocrinology)
        time_of_day = datetime.now().hour
        circadian = 15 + 10 * np.cos(2 * np.pi * (time_of_day - 8) / 24)
        
        # Estrogen suppresses HPA axis (Ochedalski et al., 2007, J Physiol Pharmacol)
        estrogen_suppression = 1.0 - (self.profile.estradiol / 400) * 0.15
        
        # Progesterone can enhance stress response
        prog_enhancement = 1.0 + (self.profile.progesterone / 30) * 0.1
        
        # Calculate cortisol
        stress['cortisol'] = circadian * estrogen_suppression * prog_enhancement
        
        # Cortisol awakening response (Pruessner et al., 1997, Psychoneuroendocrinology)
        if 6 <= time_of_day <= 8:
            stress['car'] = stress['cortisol'] * 1.5
        else:
            stress['car'] = stress['cortisol']
        
        # DHEA-S (counter-regulatory to cortisol)
        stress['dheas'] = 150 - stress['cortisol'] * 2
        
        return stress
    
    def _predict_symptoms(self, biomarkers: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict symptoms based on current biomarker state
        """
        predicted = {}
        
        # GI symptoms influenced by gut permeability and inflammation
        # Meleine & Matricon, 2014, Gastroenterology
        predicted['bloating'] = (
            biomarkers['gut_permeability'] * 3 +
            biomarkers['neuroinflammation'] * 2 +
            (1 - biomarkers['microbiome_diversity'] / 4) * 2
        )
        
        # Mood symptoms
        # Schmidt, 2012, Dialogues Clin Neurosci - hormones and mood
        serotonin_deficit = max(0, 2 - biomarkers['serotonin'])
        gaba_deficit = max(0, 100 - biomarkers['gaba'])
        
        predicted['anxiety'] = (
            serotonin_deficit * 2 +
            gaba_deficit / 20 +
            biomarkers['neuroinflammation'] * 3
        )
        
        predicted['depression'] = (
            serotonin_deficit * 3 +
            biomarkers['neuroinflammation'] * 2 +
            (20 - biomarkers['cortisol']) / 4 if biomarkers['cortisol'] < 20 else 0
        )
        
        # PMS symptoms intensified in luteal phase
        # Eisenlohr-Moul et al., 2016, Clin Psychol Sci
        if self.profile.menstrual_phase == MenstrualPhase.LUTEAL:
            pms_multiplier = 1.5
            predicted['irritability'] = predicted['anxiety'] * pms_multiplier
            predicted['mood_swings'] = (predicted['anxiety'] + predicted['depression']) / 2 * pms_multiplier
        else:
            predicted['irritability'] = predicted['anxiety'] * 0.7
            predicted['mood_swings'] = predicted['anxiety'] * 0.5
        
        # Normalize all to 0-10 scale
        for symptom in predicted:
            predicted[symptom] = min(10, max(0, predicted[symptom]))
        
        return predicted
    
    def _calculate_metabolome(self) -> Dict[str, Dict]:
        """
        Calculate microbial metabolite production
        """
        metabolites = {}
        
        # SCFAs - affected by fiber and microbiome
        # Koh et al., 2016, Cell
        metabolites['scfa'] = {
            'acetate': 60000 * self.microbiome.bacteroidetes / 30,
            'propionate': 20000 * self.microbiome.bacteroidetes / 30,
            'butyrate': 20000 * self.microbiome.firmicutes / 40
        }
        
        # Tryptophan metabolism
        # Kennedy et al., 2017, Brain Behav Immun
        metabolites['tryptophan'] = {
            'serotonin_pathway': 50 * 0.05,  # 5% via serotonin
            'kynurenine_pathway': 50 * 0.95,  # 95% via kynurenine
            'indole_pathway': 50 * self.microbiome.gram_negative_abundance / 100
        }
        
        return metabolites
    
    def recommend_intervention(self, symptom_profile: SymptomTracking) -> List[Dict[str, Any]]:
        """
        Recommend personalized interventions based on symptoms and biomarkers
        """
        recommendations = []
        
        # Analyze primary symptoms
        primary_gi = (symptom_profile.bloating + symptom_profile.abdominal_pain + 
                     symptom_profile.constipation + symptom_profile.diarrhea) / 4
        primary_mood = (symptom_profile.anxiety + symptom_profile.depression + 
                       symptom_profile.irritability) / 3
        
        if primary_gi > 5:
            # GI-focused interventions
            
            if self.microbiome.lactobacillus < 7:
                recommendations.append({
                    'type': 'probiotic',
                    'specific': 'Lactobacillus rhamnosus GG',
                    'dose': '10 billion CFU daily',
                    'evidence': 'Pedersen et al., 2019, Gut Microbes - reduces IBS symptoms',
                    'duration': '8 weeks'
                })
            
            if self.metabolome.butyrate < 15000:
                recommendations.append({
                    'type': 'prebiotic',
                    'specific': 'Inulin or Resistant Starch',
                    'dose': '10-15g daily',
                    'evidence': 'Valeur et al., 2018, Aliment Pharmacol Ther',
                    'duration': '4 weeks'
                })
        
        if primary_mood > 5:
            # Mood-focused interventions
            
            recommendations.append({
                'type': 'probiotic',
                'specific': 'Lactobacillus helveticus R0052 + Bifidobacterium longum R0175',
                'dose': '3 billion CFU daily',
                'evidence': 'Messaoudi et al., 2011, Br J Nutr - reduces anxiety',
                'duration': '8 weeks'
            })
            
            if self.profile.menstrual_phase == MenstrualPhase.LUTEAL:
                recommendations.append({
                    'type': 'supplement',
                    'specific': 'Calcium + Vitamin D',
                    'dose': '1200mg Ca + 400IU D3',
                    'evidence': 'Ghanbari et al., 2009, Taiwan J Obstet Gynecol - reduces PMS',
                    'duration': 'Continuous'
                })
        
        # Phase-specific recommendations
        if self.profile.menstrual_phase == MenstrualPhase.MENSTRUAL:
            recommendations.append({
                'type': 'dietary',
                'specific': 'Anti-inflammatory foods',
                'details': 'Omega-3 fatty acids, turmeric, ginger',
                'evidence': 'Bajalan et al., 2019, Phytother Res - reduces menstrual pain',
                'duration': 'During menstruation'
            })
        
        # Estrobolome optimization
        if self.estrobolome.beta_glucuronidase_activity > 1.5:
            recommendations.append({
                'type': 'dietary',
                'specific': 'Increase fiber to 25-30g/day',
                'details': 'Focus on soluble fiber',
                'evidence': 'Gaskins et al., 2009, Am J Clin Nutr - reduces estrogen recirculation',
                'duration': 'Continuous'
            })
        
        return recommendations

class HPAAxis:
    """
    Hypothalamic-Pituitary-Adrenal axis with sex differences
    Kudielka & Kirschbaum, 2005, Biol Psychol
    """
    
    def __init__(self):
        self.crh = 1.0  # Corticotropin-releasing hormone
        self.acth = 30.0  # Adrenocorticotropic hormone  
        self.cortisol = 12.0  # μg/dL
        self.dhea = 150.0  # μg/dL
        
    def calculate_stress_response(self, stressor: float, estradiol: float, 
                                 progesterone: float) -> Dict[str, float]:
        """
        Sex hormone modulation of stress response
        Kajantie & Phillips, 2006, Trends Endocrinol Metab
        """
        # Estrogen suppresses HPA axis
        estrogen_modulation = 1.0 - (estradiol / 400) * 0.2
        
        # Progesterone metabolites affect stress response
        prog_modulation = 1.0 + (progesterone / 30) * 0.15
        
        # Calculate response
        self.crh = stressor * estrogen_modulation * prog_modulation
        self.acth = self.crh * 30
        self.cortisol = self.acth / 2.5
        
        return {
            'crh': self.crh,
            'acth': self.acth,
            'cortisol': self.cortisol,
            'dhea': self.dhea
        }

class HPGAxis:
    """
    Hypothalamic-Pituitary-Gonadal axis
    Messinis et al., 2014, Hum Reprod Update
    """
    
    def __init__(self):
        self.gnrh_pulses_per_hour = 1.0
        self.lh = 5.0  # mIU/mL
        self.fsh = 6.0  # mIU/mL
        self.estradiol = 80.0  # pg/mL
        self.progesterone = 0.5  # ng/mL
        
    def calculate_feedback(self, estradiol: float, progesterone: float, 
                          stress_level: float) -> Dict[str, float]:
        """
        Negative and positive feedback mechanisms
        Hall, 2015, Endocrinology
        """
        # Negative feedback at moderate E2 levels
        if estradiol < 200:
            gnrh_suppression = estradiol / 200 * 0.5
            self.gnrh_pulses_per_hour = 1.0 * (1 - gnrh_suppression)
        else:
            # Positive feedback triggers LH surge
            self.gnrh_pulses_per_hour = 2.0
            self.lh = 25 + estradiol / 10
        
        # Stress suppresses HPG axis
        # Whirledge & Cidlowski, 2010, Trends Endocrinol Metab
        stress_suppression = 1.0 - min(0.5, stress_level / 20)
        self.gnrh_pulses_per_hour *= stress_suppression
        
        return {
            'gnrh_pulses': self.gnrh_pulses_per_hour,
            'lh': self.lh,
            'fsh': self.fsh
        }

class VagusNerveSystem:
    """
    Vagus nerve with sex differences in tone and reactivity
    Koenig & Thayer, 2016, Biol Sex Differ
    """
    
    def __init__(self):
        self.tone = 10.0  # Hz
        self.hrv = 50.0  # ms RMSSD
        
    def calculate_tone(self, estradiol: float, inflammation: float, 
                      scfa: Dict[str, float]) -> float:
        """
        Women typically have higher vagal tone
        """
        # Estrogen enhances vagal tone (Du et al., 2009, Auton Neurosci)
        estrogen_effect = 1.0 + (estradiol / 300) * 0.15
        
        # SCFAs enhance vagal tone (Dalile et al., 2019, Nat Rev Gastroenterol Hepatol)
        scfa_effect = 1.0 + (scfa.get('butyrate', 0) / 30000) * 0.2
        
        # Inflammation suppresses vagal tone
        inflammation_effect = 1.0 - inflammation * 0.2
        
        self.tone = 10.0 * estrogen_effect * scfa_effect * inflammation_effect
        
        return self.tone

class EntericNervousSystem:
    """
    ENS with sex hormone receptors
    Mulak & Taché, 2010, World J Gastroenterol
    """
    
    def __init__(self):
        self.motility = 1.0
        self.secretion = 1.0
        self.serotonin_ec_cells = 1.0  # Enterochromaffin cells
        
    def modulate_function(self, estradiol: float, progesterone: float) -> Dict[str, float]:
        """
        Sex hormones affect GI function
        """
        # Estrogen tends to slow transit (Gonenne et al., 2006, Am J Gastroenterol)
        self.motility = 1.0 - (estradiol / 400) * 0.2
        
        # Progesterone slows motility more (Wald et al., 1981, Gastroenterology)
        self.motility *= (1.0 - (progesterone / 30) * 0.3)
        
        # Serotonin production affected by estrogen (Krajnak, 2022, Front Neuroendocrinol)
        self.serotonin_ec_cells = 1.0 + (estradiol / 300) * 0.25
        
        return {
            'motility': self.motility,
            'secretion': self.secretion,
            'serotonin_production': self.serotonin_ec_cells
        }

class NeuroImmuneSystem:
    """
    Sex differences in neuroimmune interactions
    Klein & Flanagan, 2016, Nat Rev Immunol
    """
    
    def __init__(self):
        self.microglia_activation = 0.1
        self.cytokines = {
            'il6': 2.0,
            'tnf_alpha': 5.0,
            'il1_beta': 3.0,
            'il10': 10.0,
            'tgf_beta': 5.0
        }
        
    def calculate_neuroinflammation(self, estradiol: float, 
                                   gut_permeability: float) -> float:
        """
        Sex hormones modulate neuroinflammation
        Villa et al., 2015, Front Immunol
        """
        # Estrogen is generally anti-inflammatory in CNS
        estrogen_protection = 1.0 - (estradiol / 400) * 0.25
        
        # Gut-derived inflammation
        gut_inflammation = gut_permeability * 0.3
        
        # Microglial activation
        self.microglia_activation = (0.1 + gut_inflammation) * estrogen_protection
        
        # Calculate overall neuroinflammation
        pro_inflammatory = (self.cytokines['il6'] + self.cytokines['tnf_alpha'] + 
                           self.cytokines['il1_beta']) / 15
        anti_inflammatory = (self.cytokines['il10'] + self.cytokines['tgf_beta']) / 15
        
        neuroinflammation = (pro_inflammatory / (anti_inflammatory + 1)) * self.microglia_activation
        
        return min(1.0, neuroinflammation)

class MicrobiomeComposition:
    """
    Gut microbiome with sex-specific characteristics
    Org et al., 2016, Cell Rep
    """
    
    def __init__(self):
        # Phylum level (%)
        self.firmicutes = 45.0  # Women tend to have lower F/B ratio
        self.bacteroidetes = 35.0
        self.actinobacteria = 8.0
        self.proteobacteria = 5.0
        self.verrucomicrobia = 2.0
        
        # Key genera (log CFU/g)
        self.lactobacillus = 8.0
        self.bifidobacterium = 8.5
        self.prevotella = 7.0
        self.bacteroides = 9.0
        
        # Pathobionts
        self.gram_negative_abundance = 30.0  # % for LPS estimation
        
        self.dysbiosis_index = 0.0
        
    def calculate_dysbiosis(self) -> float:
        """
        Calculate dysbiosis index
        Shin et al., 2015, Gut Microbes
        """
        # F/B ratio (women normal ~1.3)
        fb_ratio = self.firmicutes / (self.bacteroidetes + 1)
        fb_dysbiosis = abs(fb_ratio - 1.3) / 1.3
        
        # Low beneficial bacteria
        beneficial_score = (self.lactobacillus + self.bifidobacterium) / 16
        beneficial_dysbiosis = max(0, 1 - beneficial_score)
        
        # High pathobionts
        pathobiont_dysbiosis = self.gram_negative_abundance / 100
        
        self.dysbiosis_index = (fb_dysbiosis * 0.3 + beneficial_dysbiosis * 0.4 + 
                               pathobiont_dysbiosis * 0.3)
        
        return self.dysbiosis_index

class MicrobialMetabolome:
    """
    Microbial metabolite production
    Koh et al., 2016, Cell
    """
    
    def __init__(self):
        self.scfa = {
            'acetate': 60000,  # μM
            'propionate': 20000,
            'butyrate': 20000
        }
        self.butyrate = 20000  # Separate for convenience
        self.bile_acids = {}
        self.tryptophan_metabolites = {}

class BrainRegion:
    """
    Brain region with sex hormone receptors
    McEwen & Milner, 2017, J Neurosci Res
    """
    
    def __init__(self, name: str, estrogen_receptor_alpha: bool = False,
                 estrogen_receptor_beta: bool = False,
                 progesterone_receptor: bool = False,
                 androgen_receptor: bool = False,
                 bdnf_expression: float = 1.0,
                 gnrh_neurons: int = 0):
        
        self.name = name
        self.er_alpha = estrogen_receptor_alpha
        self.er_beta = estrogen_receptor_beta
        self.pr = progesterone_receptor
        self.ar = androgen_receptor
        self.bdnf = bdnf_expression
        self.gnrh_neurons = gnrh_neurons
        self.activity = 1.0
        
    def modulate_by_hormones(self, estradiol: float, progesterone: float, 
                            testosterone: float) -> float:
        """
        Calculate hormone-modulated activity
        """
        modulation = 1.0
        
        if self.er_alpha or self.er_beta:
            # Estrogen effects (usually enhancing in hippocampus, PFC)
            estrogen_effect = 1.0 + (estradiol / 300) * 0.2
            modulation *= estrogen_effect
        
        if self.pr:
            # Progesterone effects (complex, can be anxiogenic or anxiolytic)
            prog_effect = 1.0 + (progesterone / 30) * 0.1
            modulation *= prog_effect
        
        if self.ar:
            # Androgen effects
            androgen_effect = 1.0 + (testosterone / 100) * 0.1
            modulation *= androgen_effect
        
        self.activity = modulation
        return self.activity


def run_women_health_simulation():
    """
    Run a complete menstrual cycle simulation
    """
    print("WOMEN'S HEALTH GUT-BRAIN AXIS SIMULATION")
    print("-" * 50)
    
    # Initialize model
    model = WomensGutBrainAxis()
    model.profile.age = 32
    model.profile.bmi = 24
    model.profile.cycle_length = 28
    
    print(f"Patient Profile:")
    print(f"  Age: {model.profile.age}")
    print(f"  BMI: {model.profile.bmi}")
    print(f"  Cycle Length: {model.profile.cycle_length} days")
    print()
    
    # Simulate full cycle
    cycle_data = []
    
    for day in range(1, 29):
        results = model.simulate_cycle_day(day)
        results['day'] = day
        cycle_data.append(results)
        
        # Print key days
        if day in [1, 7, 14, 21, 28]:
            print(f"Day {day} ({model.profile.menstrual_phase.name}):")
            print(f"  Estradiol: {model.profile.estradiol:.1f} pg/mL")
            print(f"  Progesterone: {model.profile.progesterone:.1f} ng/mL")
            print(f"  Gut Permeability: {results['gut_permeability']:.2f}")
            print(f"  Anxiety Score: {results['predicted_symptoms']['anxiety']:.1f}/10")
            print(f"  Bloating Score: {results['predicted_symptoms']['bloating']:.1f}/10")
            print()
    
    # Generate recommendations
    print("PERSONALIZED RECOMMENDATIONS")
    print("-" * 50)
    
    # Create symptom profile for recommendations
    test_symptoms = SymptomTracking(
        bloating=7.0,
        anxiety=6.0,
        mood_swings=5.0,
        fatigue=6.0
    )
    
    recommendations = model.recommend_intervention(test_symptoms)
    
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"{i}. {rec['type'].upper()}: {rec['specific']}")
        print(f"   Dose: {rec.get('dose', rec.get('details', 'See details'))}")
        print(f"   Evidence: {rec['evidence'][:60]}...")
        print()
    
    print("-" * 50)
    print("SIMULATION COMPLETE")
    
    return pd.DataFrame(cycle_data)


if __name__ == "__main__":
    # Run simulation
    cycle_df = run_women_health_simulation()
    
    # Additional analysis
    print("\nCYCLE ANALYSIS")
    print("-" * 50)
    print(f"Mean Gut Permeability: {cycle_df['gut_permeability'].mean():.2f}")
    print(f"Peak Anxiety Day: {cycle_df['predicted_symptoms'].apply(lambda x: x['anxiety']).idxmax() + 1}")
    print(f"Peak Bloating Day: {cycle_df['predicted_symptoms'].apply(lambda x: x['bloating']).idxmax() + 1}")

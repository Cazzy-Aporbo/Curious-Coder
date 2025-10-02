"""
INTEGRATED GUT-BRAIN AXIS RESEARCH MODEL
Multi-disciplinary computational framework for microbiome-brain interactions

This model integrates:
- Microbial metabolomics and community dynamics
- Vagus nerve signaling and enteric nervous system
- Neurotransmitter production by gut bacteria
- Immune-mediated pathways (cytokines, microglia)
- HPA axis and stress response
- Blood-brain barrier permeability
- Nutritional modulation of microbiome
- Circadian rhythm interactions
- Short-chain fatty acid signaling
- Tryptophan metabolism pathways

Based on current research from:
- Nature Reviews Gastroenterology & Hepatology
- Cell Host & Microbe
- Nature Microbiology
- Science Translational Medicine
- Gut Microbes journal

Author: Cazzy Aporbo, MS
Version: 1.0.0
Python Requirements: 3.9+
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any, Protocol
from enum import Enum, auto
import scipy.integrate as integrate
from scipy.stats import pearsonr, spearmanr
from scipy.signal import hilbert, find_peaks
from scipy.optimize import minimize, differential_evolution
import networkx as nx
from collections import defaultdict, deque
import logging
from datetime import datetime, timedelta
import json
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verified biological constants and reference ranges
class BiologicalConstants:
    """Verified reference values from literature"""
    
    # Neurotransmitter levels (nM in CSF/blood)
    SEROTONIN_CSF = 1.5  # nM
    GABA_CSF = 100  # nM
    DOPAMINE_CSF = 0.5  # nM
    NOREPINEPHRINE_CSF = 2.0  # nM
    GLUTAMATE_CSF = 10000  # nM
    
    # Short-chain fatty acids (μM in colon)
    ACETATE_COLON = 60000  # μM
    PROPIONATE_COLON = 20000  # μM
    BUTYRATE_COLON = 20000  # μM
    
    # Bacterial abundances (CFU/g feces)
    TOTAL_BACTERIA = 1e11  # CFU/g
    LACTOBACILLUS = 1e8  # CFU/g
    BIFIDOBACTERIUM = 1e9  # CFU/g
    BACTEROIDES = 1e10  # CFU/g
    
    # Cytokine baseline levels (pg/mL)
    IL1_BETA = 5  # pg/mL
    IL6 = 2  # pg/mL
    IL10 = 10  # pg/mL
    TNF_ALPHA = 8  # pg/mL
    
    # Vagus nerve parameters
    VAGUS_FIRING_RATE = 10  # Hz baseline
    VAGUS_CONDUCTION_VELOCITY = 10  # m/s
    
    # Blood-brain barrier permeability
    BBB_PERMEABILITY_NORMAL = 1e-7  # cm/s for small molecules

class MicrobialSpecies(Enum):
    """Key bacterial species affecting gut-brain axis"""
    # Beneficial psychobiotics
    LACTOBACILLUS_RHAMNOSUS = "L. rhamnosus GG"  # Reduces anxiety
    LACTOBACILLUS_HELVETICUS = "L. helveticus R0052"  # Reduces depression
    BIFIDOBACTERIUM_LONGUM = "B. longum 1714"  # Reduces stress
    BIFIDOBACTERIUM_INFANTIS = "B. infantis 35624"  # Anti-inflammatory
    
    # GABA producers
    LACTOBACILLUS_BREVIS = "L. brevis"  # High GABA production
    BIFIDOBACTERIUM_DENTIUM = "B. dentium"  # GABA synthesis
    
    # Serotonin modulators
    ENTEROCOCCUS_FAECALIS = "E. faecalis"  # Serotonin production
    STREPTOCOCCUS_THERMOPHILUS = "S. thermophilus"  # Serotonin precursors
    
    # SCFA producers
    FAECALIBACTERIUM_PRAUSNITZII = "F. prausnitzii"  # Butyrate producer
    ROSEBURIA_INTESTINALIS = "R. intestinalis"  # Butyrate producer
    AKKERMANSIA_MUCINIPHILA = "A. muciniphila"  # Mucin degrader, metabolic health
    
    # Pathobionts
    CLOSTRIDIUM_DIFFICILE = "C. difficile"  # Pathogenic
    ESCHERICHIA_COLI = "E. coli"  # Can be pathogenic
    HELICOBACTER_PYLORI = "H. pylori"  # Associated with anxiety

@dataclass
class MicrobiomeProfile:
    """Comprehensive gut microbiome characterization"""
    
    # Diversity metrics
    shannon_diversity: float = 3.5  # Typical healthy range 3-4
    simpson_index: float = 0.9
    richness: int = 500  # Number of species
    
    # Phylum-level composition (%)
    firmicutes: float = 50.0
    bacteroidetes: float = 40.0
    actinobacteria: float = 5.0
    proteobacteria: float = 3.0
    verrucomicrobia: float = 1.0
    other_phyla: float = 1.0
    
    # Key species abundances (log10 CFU/g)
    species_abundance: Dict[str, float] = field(default_factory=dict)
    
    # Functional capacity
    butyrate_production_capacity: float = 1.0  # Normalized
    gaba_production_capacity: float = 1.0
    serotonin_modulation_capacity: float = 1.0
    lps_production: float = 1.0  # Lipopolysaccharide (endotoxin)
    
    # Metabolome
    scfa_profile: Dict[str, float] = field(default_factory=dict)
    tryptophan_metabolites: Dict[str, float] = field(default_factory=dict)
    bile_acids: Dict[str, float] = field(default_factory=dict)
    
    def calculate_dysbiosis_index(self) -> float:
        """Calculate microbiome dysbiosis index"""
        # Based on Gevers et al. 2014, Ni et al. 2017
        
        # Firmicutes/Bacteroidetes ratio (normal ~1.25)
        fb_ratio = self.firmicutes / (self.bacteroidetes + 0.1)
        fb_dysbiosis = abs(fb_ratio - 1.25) / 1.25
        
        # Diversity loss
        diversity_loss = max(0, (3.5 - self.shannon_diversity) / 3.5)
        
        # Proteobacteria expansion (marker of dysbiosis)
        proteo_expansion = max(0, (self.proteobacteria - 3) / 3)
        
        # Low butyrate producers
        butyrate_deficiency = 1 - self.butyrate_production_capacity
        
        # Calculate weighted dysbiosis index
        dysbiosis = (
            fb_dysbiosis * 0.2 +
            diversity_loss * 0.3 +
            proteo_expansion * 0.25 +
            butyrate_deficiency * 0.25
        )
        
        return min(1.0, dysbiosis)

@dataclass
class NeurotransmitterLevels:
    """CNS and ENS neurotransmitter concentrations"""
    
    # Monoamines (nM)
    serotonin_gut: float = 500.0  # 90% of body's serotonin in gut
    serotonin_brain: float = 1.5
    dopamine_gut: float = 100.0
    dopamine_brain: float = 0.5
    norepinephrine: float = 2.0
    
    # Amino acid neurotransmitters (nM)
    gaba_gut: float = 1000.0
    gaba_brain: float = 100.0
    glutamate: float = 10000.0
    glycine: float = 5000.0
    
    # Neuropeptides (pM)
    substance_p: float = 50.0
    neuropeptide_y: float = 100.0
    vip: float = 30.0  # Vasoactive intestinal peptide
    cgrp: float = 20.0  # Calcitonin gene-related peptide
    
    # Gut hormones affecting brain (pM)
    ghrelin: float = 100.0
    glp1: float = 50.0  # Glucagon-like peptide 1
    pyy: float = 50.0  # Peptide YY
    cck: float = 5.0  # Cholecystokinin
    
    def calculate_excitation_inhibition_balance(self) -> float:
        """Calculate E/I balance (critical for brain function)"""
        excitation = self.glutamate + self.substance_p
        inhibition = self.gaba_brain + self.glycine
        return excitation / (inhibition + 1)

class GutBrainAxis:
    """Main integrated gut-brain axis model"""
    
    def __init__(self):
        self.microbiome = MicrobiomeProfile()
        self.neurotransmitters = NeurotransmitterLevels()
        
        # Initialize subsystems
        self.vagus_nerve = VagusNerveModel()
        self.enteric_nervous_system = EntericNervousSystem()
        self.hpa_axis = HPAAxis()
        self.immune_system = NeuroImmuneInterface()
        self.blood_brain_barrier = BloodBrainBarrier()
        self.circadian_clock = CircadianRhythm()
        
        # Metabolic pathways
        self.tryptophan_pathway = TryptophanMetabolism()
        self.scfa_signaling = SCFASignaling()
        
        # Neural networks
        self.brain_regions = self._initialize_brain_regions()
        self.gut_brain_network = self._build_network()
        
    def _initialize_brain_regions(self) -> Dict[str, 'BrainRegion']:
        """Initialize key brain regions involved in gut-brain axis"""
        regions = {}
        
        # Based on Mayer et al. 2014, Cryan & Dinan 2012
        regions['prefrontal_cortex'] = BrainRegion(
            name='Prefrontal Cortex',
            neurotransmitter_receptors={
                'serotonin': ['5HT1A', '5HT2A', '5HT2C'],
                'gaba': ['GABA_A', 'GABA_B'],
                'glutamate': ['NMDA', 'AMPA'],
                'dopamine': ['D1', 'D2']
            }
        )
        
        regions['hippocampus'] = BrainRegion(
            name='Hippocampus',
            neurotransmitter_receptors={
                'serotonin': ['5HT1A', '5HT4', '5HT7'],
                'gaba': ['GABA_A'],
                'glutamate': ['NMDA', 'AMPA', 'mGluR'],
                'bdnf': ['TrkB']  # Brain-derived neurotrophic factor
            }
        )
        
        regions['amygdala'] = BrainRegion(
            name='Amygdala',
            neurotransmitter_receptors={
                'serotonin': ['5HT1A', '5HT2C'],
                'gaba': ['GABA_A'],
                'crf': ['CRF1', 'CRF2']  # Corticotropin-releasing factor
            }
        )
        
        regions['hypothalamus'] = BrainRegion(
            name='Hypothalamus',
            neurotransmitter_receptors={
                'ghrelin': ['GHSR'],
                'leptin': ['LEPR'],
                'npy': ['Y1', 'Y2', 'Y5'],
                'crf': ['CRF1']
            }
        )
        
        regions['brainstem'] = BrainRegion(
            name='Brainstem',
            neurotransmitter_receptors={
                'serotonin': ['5HT3'],  # Important for nausea
                'substance_p': ['NK1'],
                'glutamate': ['NMDA']
            }
        )
        
        return regions
    
    def _build_network(self) -> nx.DiGraph:
        """Build the gut-brain communication network"""
        G = nx.DiGraph()
        
        # Add nodes
        G.add_node('gut_microbiome', type='microbiome')
        G.add_node('enteric_nervous_system', type='neural')
        G.add_node('vagus_nerve', type='neural')
        G.add_node('spinal_afferents', type='neural')
        G.add_node('immune_system', type='immune')
        G.add_node('hpa_axis', type='endocrine')
        G.add_node('blood_brain_barrier', type='barrier')
        
        for region in self.brain_regions:
            G.add_node(region, type='brain')
        
        # Add edges with mechanisms
        # Direct neural pathways
        G.add_edge('enteric_nervous_system', 'vagus_nerve', 
                   mechanism='afferent_signals', weight=0.8)
        G.add_edge('vagus_nerve', 'brainstem', 
                   mechanism='neural_transmission', weight=0.9)
        G.add_edge('brainstem', 'hypothalamus', 
                   mechanism='ascending_pathways', weight=0.7)
        
        # Immune pathways
        G.add_edge('gut_microbiome', 'immune_system', 
                   mechanism='pattern_recognition', weight=0.8)
        G.add_edge('immune_system', 'blood_brain_barrier', 
                   mechanism='cytokine_signaling', weight=0.6)
        G.add_edge('immune_system', 'vagus_nerve', 
                   mechanism='inflammatory_reflex', weight=0.7)
        
        # Metabolic pathways
        G.add_edge('gut_microbiome', 'blood_brain_barrier', 
                   mechanism='metabolite_transport', weight=0.5)
        
        # HPA axis
        G.add_edge('hypothalamus', 'hpa_axis', 
                   mechanism='crf_release', weight=0.9)
        G.add_edge('hpa_axis', 'gut_microbiome', 
                   mechanism='cortisol_effects', weight=0.6)
        
        return G
    
    def simulate_microbiome_intervention(self, 
                                       intervention: Dict[str, Any],
                                       duration_days: int = 30) -> pd.DataFrame:
        """
        Simulate the effects of a microbiome intervention
        (probiotic, prebiotic, dietary change, etc.)
        """
        
        results = []
        
        for day in range(duration_days):
            # Update microbiome based on intervention
            if intervention['type'] == 'probiotic':
                self._apply_probiotic(intervention['species'], 
                                     intervention['dose_cfu'])
            elif intervention['type'] == 'prebiotic':
                self._apply_prebiotic(intervention['substrate'], 
                                    intervention['dose_g'])
            elif intervention['type'] == 'diet':
                self._apply_dietary_change(intervention['diet_type'])
            
            # Calculate downstream effects
            
            # 1. Microbial metabolite production
            scfa_levels = self.scfa_signaling.calculate_production(
                self.microbiome
            )
            
            # 2. Neurotransmitter modulation
            nt_changes = self._calculate_neurotransmitter_changes()
            
            # 3. Vagus nerve activity
            vagus_activity = self.vagus_nerve.calculate_activity(
                scfa_levels, self.microbiome.lps_production
            )
            
            # 4. Neuroinflammation
            inflammation = self.immune_system.calculate_neuroinflammation(
                self.microbiome, self.blood_brain_barrier.permeability
            )
            
            # 5. HPA axis activity
            cortisol = self.hpa_axis.calculate_cortisol_rhythm(
                day, inflammation, vagus_activity
            )
            
            # 6. Brain region activity
            brain_activity = self._calculate_brain_activity()
            
            # 7. Behavioral outcomes
            behavioral_scores = self._predict_behavioral_outcomes(
                brain_activity, nt_changes, inflammation
            )
            
            results.append({
                'day': day,
                'dysbiosis_index': self.microbiome.calculate_dysbiosis_index(),
                'shannon_diversity': self.microbiome.shannon_diversity,
                'butyrate': scfa_levels['butyrate'],
                'acetate': scfa_levels['acetate'],
                'propionate': scfa_levels['propionate'],
                'vagus_tone': vagus_activity,
                'neuroinflammation': inflammation,
                'cortisol': cortisol,
                'serotonin_brain': nt_changes['serotonin'],
                'gaba_brain': nt_changes['gaba'],
                'anxiety_score': behavioral_scores['anxiety'],
                'depression_score': behavioral_scores['depression'],
                'cognitive_score': behavioral_scores['cognition'],
                'gi_symptoms': behavioral_scores['gi_symptoms']
            })
        
        return pd.DataFrame(results)
    
    def _apply_probiotic(self, species: str, dose_cfu: float):
        """Apply probiotic intervention to microbiome"""
        
        # Colonization dynamics based on Derrien & van Hylckama Vlieg 2015
        colonization_rate = 0.1  # 10% colonization efficiency
        
        if 'lactobacillus' in species.lower():
            current = self.microbiome.species_abundance.get(species, 6)
            administered = np.log10(dose_cfu)
            new_abundance = np.log10(10**current + 10**administered * colonization_rate)
            self.microbiome.species_abundance[species] = new_abundance
            
            # Increase GABA production capacity
            self.microbiome.gaba_production_capacity *= 1.05
            
            # Reduce pathobiont abundance (competitive exclusion)
            self.microbiome.lps_production *= 0.95
            
        elif 'bifidobacterium' in species.lower():
            current = self.microbiome.species_abundance.get(species, 7)
            administered = np.log10(dose_cfu)
            new_abundance = np.log10(10**current + 10**administered * colonization_rate)
            self.microbiome.species_abundance[species] = new_abundance
            
            # Increase butyrate production
            self.microbiome.butyrate_production_capacity *= 1.03
            
            # Improve barrier function
            self.blood_brain_barrier.permeability *= 0.98
    
    def _apply_prebiotic(self, substrate: str, dose_g: float):
        """Apply prebiotic intervention"""
        
        if substrate == 'inulin':
            # Selectively feed Bifidobacterium
            self.microbiome.actinobacteria *= 1.02
            self.microbiome.butyrate_production_capacity *= 1.04
            
        elif substrate == 'fos':  # Fructooligosaccharides
            # Increase SCFA production
            self.microbiome.scfa_profile['acetate'] *= 1.1
            self.microbiome.scfa_profile['butyrate'] *= 1.05
            
        elif substrate == 'gos':  # Galactooligosaccharides
            # Increase Bifidobacterium
            self.microbiome.actinobacteria *= 1.03
            # Reduce proteobacteria
            self.microbiome.proteobacteria *= 0.97
    
    def _apply_dietary_change(self, diet_type: str):
        """Apply dietary intervention effects"""
        
        if diet_type == 'mediterranean':
            # Based on De Filippis et al. 2016
            self.microbiome.shannon_diversity *= 1.01
            self.microbiome.firmicutes *= 0.99
            self.microbiome.bacteroidetes *= 1.01
            self.microbiome.butyrate_production_capacity *= 1.02
            
        elif diet_type == 'high_fiber':
            # Based on Sonnenburg et al. 2016
            self.microbiome.shannon_diversity *= 1.02
            self.microbiome.butyrate_production_capacity *= 1.05
            self.microbiome.scfa_profile['butyrate'] *= 1.1
            
        elif diet_type == 'western':
            # High fat, low fiber
            self.microbiome.shannon_diversity *= 0.99
            self.microbiome.firmicutes *= 1.01
            self.microbiome.proteobacteria *= 1.02
            self.microbiome.lps_production *= 1.05
    
    def _calculate_neurotransmitter_changes(self) -> Dict[str, float]:
        """Calculate changes in brain neurotransmitters based on gut signals"""
        
        changes = {}
        
        # Serotonin: 90% produced in gut, modulated by microbiome
        # Based on Yano et al. 2015, O'Mahony et al. 2015
        tph1_activity = self.microbiome.serotonin_modulation_capacity
        spore_formers = self.microbiome.species_abundance.get(
            'clostridium_sporogenes', 6
        )
        serotonin_synthesis = tph1_activity * (1 + spore_formers/10)
        
        # Account for vagal signaling of gut serotonin
        vagal_serotonin_signal = self.vagus_nerve.serotonin_signaling()
        
        changes['serotonin'] = self.neurotransmitters.serotonin_brain * (
            0.7 + 0.2 * serotonin_synthesis + 0.1 * vagal_serotonin_signal
        )
        
        # GABA: Produced by specific bacteria
        # Based on Barrett et al. 2012, Bravo et al. 2011
        gaba_producers = [
            'lactobacillus_brevis',
            'bifidobacterium_dentium',
            'lactobacillus_rhamnosus'
        ]
        
        gaba_production = sum([
            self.microbiome.species_abundance.get(sp, 6) / 10
            for sp in gaba_producers
        ])
        
        # GABA doesn't cross BBB easily, but affects vagus nerve
        vagal_gaba_signal = self.vagus_nerve.gaba_signaling(gaba_production)
        
        changes['gaba'] = self.neurotransmitters.gaba_brain * (
            0.8 + 0.2 * vagal_gaba_signal
        )
        
        # Dopamine: Modulated by gut bacteria
        # Based on Gonzalez-Arancibia et al. 2019
        changes['dopamine'] = self.neurotransmitters.dopamine_brain * (
            0.9 + 0.1 * self.microbiome.shannon_diversity / 3.5
        )
        
        # Glutamate: Affected by gut inflammation
        inflammation = self.immune_system.calculate_neuroinflammation(
            self.microbiome, self.blood_brain_barrier.permeability
        )
        changes['glutamate'] = self.neurotransmitters.glutamate * (
            1 + 0.2 * inflammation
        )
        
        return changes
    
    def _calculate_brain_activity(self) -> Dict[str, float]:
        """Calculate activity in different brain regions"""
        
        activity = {}
        
        for region_name, region in self.brain_regions.items():
            # Base activity modulated by neurotransmitters and inflammation
            base_activity = 1.0
            
            # Neurotransmitter effects
            if 'serotonin' in region.neurotransmitter_receptors:
                serotonin_effect = (
                    self.neurotransmitters.serotonin_brain / 
                    BiologicalConstants.SEROTONIN_CSF
                )
                base_activity *= (0.8 + 0.2 * serotonin_effect)
            
            if 'gaba' in region.neurotransmitter_receptors:
                ei_balance = self.neurotransmitters.calculate_excitation_inhibition_balance()
                base_activity *= (2 / (1 + ei_balance))  # Higher GABA = lower activity
            
            # Inflammation effects (especially in hippocampus)
            if region_name == 'hippocampus':
                inflammation = self.immune_system.neuroinflammation_level
                base_activity *= (1 - 0.3 * inflammation)
            
            # Stress effects (especially in amygdala)
            if region_name == 'amygdala':
                cortisol = self.hpa_axis.cortisol_level
                base_activity *= (1 + 0.2 * cortisol / 15)  # Increased with stress
            
            activity[region_name] = base_activity
        
        return activity
    
    def _predict_behavioral_outcomes(self, 
                                    brain_activity: Dict[str, float],
                                    nt_changes: Dict[str, float],
                                    inflammation: float) -> Dict[str, float]:
        """Predict behavioral and clinical outcomes"""
        
        outcomes = {}
        
        # Anxiety score (0-10 scale)
        # High amygdala activity + low GABA + inflammation
        anxiety = (
            3 * brain_activity.get('amygdala', 1) +
            2 * (1 - nt_changes['gaba'] / BiologicalConstants.GABA_CSF) +
            2 * inflammation +
            1 * (1 - nt_changes['serotonin'] / BiologicalConstants.SEROTONIN_CSF)
        ) / 0.8  # Normalize to 0-10
        outcomes['anxiety'] = min(10, max(0, anxiety))
        
        # Depression score (0-10 scale)
        # Low PFC activity + low serotonin + low BDNF
        depression = (
            3 * (2 - brain_activity.get('prefrontal_cortex', 1)) +
            3 * (1 - nt_changes['serotonin'] / BiologicalConstants.SEROTONIN_CSF) +
            2 * (1 - brain_activity.get('hippocampus', 1)) +
            1 * inflammation
        ) / 0.9
        outcomes['depression'] = min(10, max(0, depression))
        
        # Cognitive function (0-10 scale, higher is better)
        cognition = (
            3 * brain_activity.get('prefrontal_cortex', 1) +
            3 * brain_activity.get('hippocampus', 1) +
            2 * (nt_changes['dopamine'] / BiologicalConstants.DOPAMINE_CSF) +
            -2 * inflammation
        ) / 0.8
        outcomes['cognition'] = min(10, max(0, cognition))
        
        # GI symptoms (0-10 scale)
        gi_symptoms = (
            2 * self.microbiome.calculate_dysbiosis_index() * 10 +
            2 * self.microbiome.lps_production +
            1 * abs(self.enteric_nervous_system.motility - 1) * 5 +
            1 * (10 - self.vagus_nerve.tone)
        ) / 0.6
        outcomes['gi_symptoms'] = min(10, max(0, gi_symptoms))
        
        return outcomes

class VagusNerveModel:
    """Model of vagus nerve signaling in gut-brain axis"""
    
    def __init__(self):
        self.tone = 10  # Hz, baseline firing rate
        self.afferent_signals = {}
        self.efferent_signals = {}
        
    def calculate_activity(self, scfa_levels: Dict[str, float], 
                          lps: float) -> float:
        """Calculate vagus nerve activity based on gut signals"""
        
        # SCFAs increase vagal tone (based on Goehler et al. 2005)
        scfa_effect = (
            scfa_levels.get('butyrate', 0) / 20000 * 0.3 +
            scfa_levels.get('propionate', 0) / 20000 * 0.2 +
            scfa_levels.get('acetate', 0) / 60000 * 0.1
        )
        
        # LPS decreases vagal tone (based on Ghia et al. 2006)
        lps_effect = -lps * 0.2
        
        # Calculate new tone
        self.tone = BiologicalConstants.VAGUS_FIRING_RATE * (
            1 + scfa_effect + lps_effect
        )
        
        return self.tone
    
    def serotonin_signaling(self) -> float:
        """5-HT3 receptor activation on vagal afferents"""
        # Based on Browning et al. 2017
        return self.afferent_signals.get('serotonin', 0.5)
    
    def gaba_signaling(self, gaba_production: float) -> float:
        """GABA signaling through vagus nerve"""
        # Based on Bravo et al. 2011 (L. rhamnosus effects)
        return min(1.0, gaba_production / 10)

class EntericNervousSystem:
    """Model of the enteric nervous system ('second brain')"""
    
    def __init__(self):
        self.neurons = {
            'sensory': 1e7,  # Intrinsic primary afferent neurons
            'motor': 2e7,  # Motor neurons
            'interneurons': 1e8  # Interneurons
        }
        self.motility = 1.0  # Normalized GI motility
        self.secretion = 1.0  # Normalized secretion
        
    def process_microbial_signals(self, microbiome: MicrobiomeProfile) -> Dict[str, float]:
        """Process signals from microbiome"""
        
        outputs = {}
        
        # Serotonin from enterochromaffin cells
        # 90% of body's serotonin is in gut
        ec_cell_serotonin = 500 * microbiome.serotonin_modulation_capacity
        outputs['serotonin'] = ec_cell_serotonin
        
        # Motility regulation
        self.motility = 1.0
        if microbiome.calculate_dysbiosis_index() > 0.5:
            # Dysbiosis can cause motility issues
            self.motility *= np.random.choice([0.7, 1.3])  # Hypo or hypermotility
        
        outputs['motility'] = self.motility
        
        return outputs

class HPAAxis:
    """Hypothalamic-pituitary-adrenal axis model"""
    
    def __init__(self):
        self.crh_level = 1.0  # Corticotropin-releasing hormone
        self.acth_level = 30.0  # Adrenocorticotropic hormone
        self.cortisol_level = 15.0  # μg/dL
        
    def calculate_cortisol_rhythm(self, time_hours: float, 
                                 inflammation: float, 
                                 vagus_tone: float) -> float:
        """Calculate cortisol with circadian rhythm and modulation"""
        
        # Circadian rhythm (peaks in morning)
        circadian = 15 + 10 * np.cos(2 * np.pi * (time_hours - 8) / 24)
        
        # Inflammation increases HPA activity
        inflammation_effect = 1 + 0.3 * inflammation
        
        # Vagus nerve inhibits HPA axis
        vagus_effect = 1 - 0.1 * (vagus_tone / 10 - 1)
        
        self.cortisol_level = circadian * inflammation_effect * vagus_effect
        
        return self.cortisol_level

class NeuroImmuneInterface:
    """Model of neuroimmune interactions in gut-brain axis"""
    
    def __init__(self):
        self.cytokines = {
            'il1_beta': BiologicalConstants.IL1_BETA,
            'il6': BiologicalConstants.IL6,
            'il10': BiologicalConstants.IL10,
            'tnf_alpha': BiologicalConstants.TNF_ALPHA
        }
        self.microglia_activation = 0.1  # 10% baseline activation
        self.neuroinflammation_level = 0.0
        
    def calculate_neuroinflammation(self, microbiome: MicrobiomeProfile, 
                                   bbb_permeability: float) -> float:
        """Calculate neuroinflammation based on gut signals"""
        
        # LPS-induced inflammation (based on Qin et al. 2007)
        lps_inflammation = microbiome.lps_production * bbb_permeability * 100
        
        # Calculate inflammatory index
        pro_inflammatory = (
            self.cytokines['il1_beta'] / 5 +
            self.cytokines['il6'] / 2 +
            self.cytokines['tnf_alpha'] / 8
        ) / 3
        
        anti_inflammatory = self.cytokines['il10'] / 10
        
        inflammation_ratio = pro_inflammatory / (anti_inflammatory + 1)
        
        # Microglial activation
        self.microglia_activation = min(1.0, 0.1 + lps_inflammation * 0.5)
        
        # Overall neuroinflammation
        self.neuroinflammation_level = (
            0.4 * inflammation_ratio +
            0.4 * self.microglia_activation +
            0.2 * lps_inflammation
        )
        
        return min(1.0, self.neuroinflammation_level)

class BloodBrainBarrier:
    """Model of blood-brain barrier permeability"""
    
    def __init__(self):
        self.permeability = BiologicalConstants.BBB_PERMEABILITY_NORMAL
        self.tight_junction_integrity = 1.0
        
    def update_permeability(self, inflammation: float, 
                           microbiome: MicrobiomeProfile):
        """Update BBB permeability based on conditions"""
        
        # Inflammation increases permeability (based on Varatharaj & Galea 2017)
        inflammation_effect = 1 + inflammation * 2
        
        # Butyrate strengthens BBB (based on Braniste et al. 2014)
        butyrate_protection = 1 - 0.2 * microbiome.butyrate_production_capacity
        
        # LPS disrupts BBB
        lps_disruption = 1 + 0.5 * microbiome.lps_production
        
        self.permeability = (
            BiologicalConstants.BBB_PERMEABILITY_NORMAL * 
            inflammation_effect * butyrate_protection * lps_disruption
        )
        
        # Update tight junction integrity
        self.tight_junction_integrity = 1 / (1 + self.permeability * 1e7)

class TryptophanMetabolism:
    """Model of tryptophan metabolism pathways"""
    
    def __init__(self):
        self.tryptophan_level = 50  # μM in blood
        
        # Three main pathways
        self.serotonin_pathway = 0.05  # 5% of tryptophan
        self.kynurenine_pathway = 0.95  # 95% of tryptophan
        self.indole_pathway = 0.0  # Bacterial metabolism
        
    def calculate_metabolites(self, microbiome: MicrobiomeProfile) -> Dict[str, float]:
        """Calculate tryptophan metabolite levels"""
        
        metabolites = {}
        
        # Serotonin pathway (based on O'Mahony et al. 2015)
        tph_activity = microbiome.serotonin_modulation_capacity
        metabolites['serotonin'] = self.tryptophan_level * 0.05 * tph_activity
        metabolites['5hiaa'] = metabolites['serotonin'] * 0.8  # 5-HIAA
        
        # Kynurenine pathway (based on Kennedy et al. 2017)
        ido_activity = 1.0  # Indoleamine 2,3-dioxygenase
        if microbiome.lps_production > 1.5:
            ido_activity *= 1.5  # LPS activates IDO
        
        metabolites['kynurenine'] = self.tryptophan_level * 0.95 * ido_activity
        metabolites['kynurenic_acid'] = metabolites['kynurenine'] * 0.3
        metabolites['quinolinic_acid'] = metabolites['kynurenine'] * 0.2
        
        # Indole pathway (bacterial, based on Roager & Licht 2018)
        indole_producers = microbiome.species_abundance.get('e_coli', 8)
        metabolites['indole'] = self.tryptophan_level * (indole_producers / 10) * 0.1
        metabolites['ipa'] = metabolites['indole'] * 0.5  # Indolepropionic acid
        
        return metabolites

class SCFASignaling:
    """Short-chain fatty acid production and signaling"""
    
    def __init__(self):
        self.receptors = {
            'GPR41': 1.0,  # Free fatty acid receptor 3
            'GPR43': 1.0,  # Free fatty acid receptor 2
            'GPR109A': 1.0  # Niacin receptor 1
        }
        
    def calculate_production(self, microbiome: MicrobiomeProfile) -> Dict[str, float]:
        """Calculate SCFA production from fiber fermentation"""
        
        scfa = {}
        
        # Based on Rios-Covian et al. 2016
        # Acetate (60% of total SCFA)
        acetate_producers = (
            microbiome.bacteroidetes * 0.4 +
            microbiome.firmicutes * 0.3
        )
        scfa['acetate'] = 60000 * (acetate_producers / 90) * microbiome.shannon_diversity / 3.5
        
        # Propionate (20% of total SCFA)
        propionate_producers = microbiome.bacteroidetes * 0.4
        scfa['propionate'] = 20000 * (propionate_producers / 40)
        
        # Butyrate (20% of total SCFA)
        scfa['butyrate'] = 20000 * microbiome.butyrate_production_capacity
        
        return scfa
    
    def calculate_signaling_effects(self, scfa_levels: Dict[str, float]) -> Dict[str, float]:
        """Calculate downstream signaling effects of SCFAs"""
        
        effects = {}
        
        # Anti-inflammatory effects (based on Vinolo et al. 2011)
        butyrate_antiinflam = scfa_levels['butyrate'] / 20000
        effects['inflammation_reduction'] = butyrate_antiinflam * 0.3
        
        # Gut barrier function (based on Peng et al. 2009)
        effects['barrier_enhancement'] = butyrate_antiinflam * 0.25
        
        # Appetite regulation via PYY and GLP-1 (based on Chambers et al. 2015)
        effects['satiety_signaling'] = (
            scfa_levels['acetate'] / 60000 * 0.2 +
            scfa_levels['propionate'] / 20000 * 0.3
        )
        
        return effects

class CircadianRhythm:
    """Model of circadian rhythm interactions with gut-brain axis"""
    
    def __init__(self):
        self.phase = 0  # Current phase (0-24 hours)
        self.amplitude = 1.0
        self.melatonin_level = 10  # pg/mL
        
    def update_phase(self, hours: float):
        """Update circadian phase"""
        self.phase = hours % 24
        
        # Calculate melatonin (peaks at night)
        if 22 <= self.phase or self.phase <= 6:
            self.melatonin_level = 50 + 30 * np.sin(
                np.pi * (self.phase - 22) / 8 if self.phase >= 22 
                else np.pi * (self.phase + 2) / 8
            )
        else:
            self.melatonin_level = 10
    
    def microbiome_oscillations(self) -> Dict[str, float]:
        """Model circadian oscillations in microbiome"""
        
        # Based on Thaiss et al. 2014, Liang et al. 2015
        oscillations = {}
        
        # Firmicutes peak during active phase
        oscillations['firmicutes_rhythm'] = 1 + 0.1 * np.sin(
            2 * np.pi * (self.phase - 20) / 24
        )
        
        # Bacteroidetes peak during rest phase
        oscillations['bacteroidetes_rhythm'] = 1 + 0.1 * np.sin(
            2 * np.pi * (self.phase - 8) / 24
        )
        
        return oscillations

class BrainRegion:
    """Model of specific brain regions affected by gut signals"""
    
    def __init__(self, name: str, neurotransmitter_receptors: Dict[str, List[str]]):
        self.name = name
        self.neurotransmitter_receptors = neurotransmitter_receptors
        self.activity_level = 1.0
        self.connectivity = {}
        
    def process_signals(self, signals: Dict[str, float]) -> float:
        """Process incoming signals and update activity"""
        
        modulation = 0
        
        for nt, receptors in self.neurotransmitter_receptors.items():
            if nt in signals:
                # Different receptors have different effects
                for receptor in receptors:
                    if '5HT1A' in receptor:  # Inhibitory serotonin receptor
                        modulation -= signals[nt] * 0.1
                    elif '5HT2' in receptor:  # Excitatory serotonin receptor
                        modulation += signals[nt] * 0.1
                    elif 'GABA' in receptor:  # Inhibitory
                        modulation -= signals[nt] * 0.2
                    elif 'NMDA' in receptor or 'AMPA' in receptor:  # Excitatory
                        modulation += signals[nt] * 0.15
        
        self.activity_level = max(0.1, min(2.0, 1 + modulation))
        return self.activity_level


def run_clinical_simulation():
    """Run a clinical simulation of gut-brain interventions"""
    
    print("GUT-BRAIN AXIS CLINICAL SIMULATION")
    print("=" * 50)
    
    # Initialize model
    model = GutBrainAxis()
    
    # Simulate different interventions
    interventions = [
        {
            'name': 'Lactobacillus rhamnosus GG Probiotic',
            'type': 'probiotic',
            'species': 'lactobacillus_rhamnosus',
            'dose_cfu': 1e10  # 10 billion CFU
        },
        {
            'name': 'High-Fiber Prebiotic (Inulin)',
            'type': 'prebiotic',
            'substrate': 'inulin',
            'dose_g': 10
        },
        {
            'name': 'Mediterranean Diet',
            'type': 'diet',
            'diet_type': 'mediterranean'
        }
    ]
    
    for intervention in interventions:
        print(f"\nSimulating: {intervention['name']}")
        print("-" * 40)
        
        results = model.simulate_microbiome_intervention(
            intervention, 
            duration_days=30
        )
        
        # Analyze results
        print(f"Day 0 -> Day 30 changes:")
        print(f"  Dysbiosis Index: {results.iloc[0]['dysbiosis_index']:.2f} -> "
              f"{results.iloc[-1]['dysbiosis_index']:.2f}")
        print(f"  Anxiety Score: {results.iloc[0]['anxiety_score']:.1f} -> "
              f"{results.iloc[-1]['anxiety_score']:.1f}")
        print(f"  Depression Score: {results.iloc[0]['depression_score']:.1f} -> "
              f"{results.iloc[-1]['depression_score']:.1f}")
        print(f"  Cognitive Score: {results.iloc[0]['cognitive_score']:.1f} -> "
              f"{results.iloc[-1]['cognitive_score']:.1f}")
        print(f"  GI Symptoms: {results.iloc[0]['gi_symptoms']:.1f} -> "
              f"{results.iloc[-1]['gi_symptoms']:.1f}")
        print(f"  Butyrate Level: {results.iloc[0]['butyrate']:.0f} -> "
              f"{results.iloc[-1]['butyrate']:.0f} μM")
        
        # Calculate effect sizes
        anxiety_change = results.iloc[-1]['anxiety_score'] - results.iloc[0]['anxiety_score']
        depression_change = results.iloc[-1]['depression_score'] - results.iloc[0]['depression_score']
        
        print(f"\nClinical Significance:")
        if abs(anxiety_change) > 2:
            print(f"  ✓ Clinically significant anxiety change: {anxiety_change:.1f} points")
        if abs(depression_change) > 2:
            print(f"  ✓ Clinically significant depression change: {depression_change:.1f} points")
    
    print("\n" + "=" * 50)
    print("SIMULATION COMPLETE")
    
    # Additional mechanistic analysis
    print("\nMECHANISTIC INSIGHTS")
    print("-" * 40)
    
    # Check tryptophan metabolism
    tryp_metabolism = TryptophanMetabolism()
    metabolites = tryp_metabolism.calculate_metabolites(model.microbiome)
    
    print("Tryptophan Metabolites:")
    print(f"  Serotonin: {metabolites['serotonin']:.1f} μM")
    print(f"  Kynurenine: {metabolites['kynurenine']:.1f} μM")
    print(f"  Kynurenic Acid: {metabolites['kynurenic_acid']:.1f} μM")
    print(f"  Indole: {metabolites['indole']:.1f} μM")
    
    # Check neuroinflammation
    inflammation = model.immune_system.calculate_neuroinflammation(
        model.microbiome, 
        model.blood_brain_barrier.permeability
    )
    print(f"\nNeuroinflammation Level: {inflammation:.2f} (0-1 scale)")
    print(f"Microglial Activation: {model.immune_system.microglia_activation:.1%}")
    
    # Check vagus nerve
    print(f"\nVagus Nerve Tone: {model.vagus_nerve.tone:.1f} Hz")
    print(f"  (Normal: {BiologicalConstants.VAGUS_FIRING_RATE} Hz)")


if __name__ == "__main__":
    run_clinical_simulation()

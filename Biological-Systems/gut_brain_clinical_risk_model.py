"""
GUT-BRAIN CLINICAL RISK ASSESSMENT AND TRIAGE MODEL
Comprehensive Decision Support System for Healthcare Providers

This model provides evidence-based risk stratification for gut-brain axis symptoms
and generates clinical recommendations based on symptom patterns, triggers, and risk factors.

IMPORTANT: This model is designed to support, not replace, clinical judgment.
All recommendations should be validated by qualified healthcare providers.

Based on:
- Rome IV Diagnostic Criteria (Drossman & Hasler, 2016, Gastroenterology)
- Red Flag Symptoms (Begtrup et al., 2013, BMC Gastroenterol)
- NICE Guidelines for IBS and Functional GI Disorders
- ACG Clinical Guidelines (Lacy et al., 2021, Am J Gastroenterol)

Version: 1.0.0 ...incomplete version..research in progress 10/1/2025
Author: Cazandra Aporbo
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from enum import Enum, auto
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, Counter
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """
    Risk stratification levels with clinical action thresholds
    Based on Emergency Severity Index (ESI) and clinical guidelines
    """
    LOW = auto()  # Self-care appropriate
    MEDIUM = auto()  # Primary care within days/weeks
    HIGH = auto()  # Urgent care or same-day appointment
    VERY_HIGH = auto()  # Emergency department immediately
    CRITICAL = auto()  # Call 911/emergency services
    
    @property
    def action_timeframe(self) -> str:
        timeframes = {
            RiskLevel.LOW: "Monitor at home, follow up if worsening",
            RiskLevel.MEDIUM: "Schedule appointment within 1-2 weeks",
            RiskLevel.HIGH: "See healthcare provider within 24-48 hours",
            RiskLevel.VERY_HIGH: "Go to emergency department now",
            RiskLevel.CRITICAL: "Call emergency services (911) immediately"
        }
        return timeframes[self]
    
    @property
    def color_code(self) -> str:
        colors = {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.HIGH: "orange",
            RiskLevel.VERY_HIGH: "red",
            RiskLevel.CRITICAL: "red_flashing"
        }
        return colors[self]

class SymptomCategory(Enum):
    """Categorization of symptoms by system"""
    GASTROINTESTINAL = auto()
    NEUROLOGICAL = auto()
    PSYCHOLOGICAL = auto()
    SYSTEMIC = auto()
    AUTONOMIC = auto()
    METABOLIC = auto()

@dataclass
class Symptom:
    """
    Individual symptom with clinical characteristics
    """
    name: str
    category: SymptomCategory
    severity: float  # 0-10 scale
    duration_days: float
    onset: str  # "sudden", "gradual"
    frequency: str  # "constant", "intermittent", "episodic"
    associated_features: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    
    def calculate_acuity_score(self) -> float:
        """
        Calculate symptom acuity based on clinical parameters
        Higher scores indicate more urgent symptoms
        """
        score = 0.0
        
        # Severity contribution
        score += self.severity * 2
        
        # Duration factors
        if self.duration_days < 1 and self.severity > 7:
            score += 5  # Acute severe symptoms
        elif self.duration_days > 14:
            score += 2  # Chronic symptoms may need evaluation
        
        # Onset type
        if self.onset == "sudden" and self.severity > 6:
            score += 3
        
        # Red flags dramatically increase score
        score += len(self.red_flags) * 10
        
        return score

@dataclass
class Trigger:
    """
    Potential trigger for symptoms
    """
    type: str  # "dietary", "stress", "medication", "infection", "hormonal"
    name: str
    timing: str  # "immediate", "hours", "days"
    strength: float  # 0-1, likelihood of causation
    evidence_level: str  # "strong", "moderate", "weak"
    
class GutBrainInteraction:
    """
    Defines specific gut-brain axis interactions and their clinical significance
    """
    
    def __init__(self):
        self.interactions = self._define_interactions()
        
    def _define_interactions(self) -> Dict[str, Dict]:
        """
        Define known gut-brain interactions with clinical relevance
        Based on Mayer et al., 2022, Physiol Rev
        """
        return {
            'inflammation': {
                'pathways': ['cytokine signaling', 'vagal afferents', 'immune activation'],
                'symptoms': ['abdominal pain', 'fatigue', 'brain fog', 'mood changes'],
                'biomarkers': ['CRP', 'calprotectin', 'IL-6'],
                'clinical_significance': 'high'
            },
            'dysbiosis': {
                'pathways': ['metabolite production', 'barrier dysfunction', 'immune dysregulation'],
                'symptoms': ['bloating', 'irregular bowel habits', 'food intolerances'],
                'biomarkers': ['dysbiosis index', 'SIBO breath test'],
                'clinical_significance': 'moderate'
            },
            'vagal_dysfunction': {
                'pathways': ['autonomic dysregulation', 'motility disorders', 'inflammation'],
                'symptoms': ['gastroparesis', 'nausea', 'early satiety', 'palpitations'],
                'biomarkers': ['HRV', 'gastric emptying study'],
                'clinical_significance': 'high'
            },
            'hpa_dysregulation': {
                'pathways': ['stress response', 'cortisol rhythm', 'gut permeability'],
                'symptoms': ['IBS symptoms', 'anxiety', 'sleep disturbance'],
                'biomarkers': ['cortisol', 'ACTH'],
                'clinical_significance': 'moderate'
            },
            'serotonin_imbalance': {
                'pathways': ['gut motility', 'mood regulation', 'visceral sensitivity'],
                'symptoms': ['constipation/diarrhea', 'depression', 'anxiety'],
                'biomarkers': ['5-HIAA', 'platelet serotonin'],
                'clinical_significance': 'moderate'
            },
            'barrier_dysfunction': {
                'pathways': ['increased permeability', 'endotoxemia', 'inflammation'],
                'symptoms': ['food sensitivities', 'fatigue', 'joint pain'],
                'biomarkers': ['zonulin', 'LPS', 'lactulose/mannitol'],
                'clinical_significance': 'moderate'
            }
        }

class ClinicalRiskAssessment:
    """
    Main clinical risk assessment engine
    """
    
    def __init__(self):
        self.red_flag_database = self._initialize_red_flags()
        self.differential_diagnoses = self._initialize_differentials()
        self.interaction_model = GutBrainInteraction()
        self.triage_criteria = self._initialize_triage_criteria()
        
    def _initialize_red_flags(self) -> Dict[str, List[str]]:
        """
        Red flag symptoms requiring urgent evaluation
        Based on NICE Guidelines and ACG recommendations
        """
        return {
            'gastrointestinal': [
                'blood in stool (melena or hematochezia)',
                'persistent vomiting with dehydration',
                'severe abdominal pain with fever',
                'signs of intestinal obstruction',
                'unexplained weight loss >10% in 3 months',
                'persistent diarrhea with dehydration',
                'jaundice',
                'abdominal mass',
                'dysphagia (difficulty swallowing)',
                'family history of GI cancer with new symptoms'
            ],
            'neurological': [
                'sudden severe headache (thunderclap)',
                'confusion or altered mental status',
                'seizures',
                'focal neurological deficits',
                'loss of consciousness',
                'severe vertigo with neurological signs'
            ],
            'systemic': [
                'fever >103°F (39.4°C) with GI symptoms',
                'signs of sepsis',
                'severe dehydration',
                'chest pain with GI symptoms',
                'shortness of breath',
                'signs of anaphylaxis'
            ],
            'metabolic': [
                'signs of diabetic ketoacidosis',
                'severe electrolyte imbalance symptoms',
                'signs of thyroid storm'
            ]
        }
    
    def _initialize_differentials(self) -> Dict[str, List[Dict]]:
        """
        Differential diagnoses to consider based on symptom patterns
        """
        return {
            'acute_abdominal_pain': [
                {'diagnosis': 'appendicitis', 'urgency': 'very_high', 'key_features': ['RLQ pain', 'fever', 'rebound tenderness']},
                {'diagnosis': 'cholecystitis', 'urgency': 'high', 'key_features': ['RUQ pain', 'fever', 'Murphy sign']},
                {'diagnosis': 'pancreatitis', 'urgency': 'high', 'key_features': ['epigastric pain', 'radiation to back', 'elevated lipase']},
                {'diagnosis': 'intestinal obstruction', 'urgency': 'very_high', 'key_features': ['distension', 'vomiting', 'absent bowel sounds']},
                {'diagnosis': 'perforation', 'urgency': 'critical', 'key_features': ['rigid abdomen', 'severe pain', 'shock']}
            ],
            'chronic_symptoms': [
                {'diagnosis': 'IBS', 'urgency': 'low', 'key_features': ['recurrent pain', 'bowel habit changes', 'no red flags']},
                {'diagnosis': 'IBD', 'urgency': 'medium', 'key_features': ['diarrhea', 'blood', 'weight loss', 'fatigue']},
                {'diagnosis': 'SIBO', 'urgency': 'low', 'key_features': ['bloating', 'gas', 'diarrhea', 'malabsorption']},
                {'diagnosis': 'functional dyspepsia', 'urgency': 'low', 'key_features': ['epigastric pain', 'early satiety', 'postprandial fullness']}
            ]
        }
    
    def _initialize_triage_criteria(self) -> Dict[str, Dict]:
        """
        Triage criteria based on validated emergency medicine protocols
        """
        return {
            'vital_signs': {
                'critical': {
                    'heart_rate': ('>140 or <40', '<50 with symptoms'),
                    'blood_pressure': ('SBP <90 or >200', 'DBP >120'),
                    'respiratory_rate': ('>30 or <8', 'labored breathing'),
                    'temperature': ('>104°F or <95°F', None),
                    'oxygen_saturation': ('<90%', None)
                }
            },
            'pain_scores': {
                'mild': (1, 3),
                'moderate': (4, 6),
                'severe': (7, 9),
                'critical': (10, 10)
            }
        }
    
    def assess_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive patient assessment returning risk level and recommendations
        """
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'risk_level': None,
            'confidence': 0.0,
            'primary_concerns': [],
            'differential_diagnoses': [],
            'recommended_actions': [],
            'follow_up': [],
            'education': []
        }
        
        # Extract symptoms
        symptoms = self._parse_symptoms(patient_data.get('symptoms', []))
        
        # Check for red flags first
        red_flags = self._check_red_flags(symptoms, patient_data)
        if red_flags:
            assessment['risk_level'] = RiskLevel.VERY_HIGH
            assessment['primary_concerns'] = red_flags
            assessment['recommended_actions'] = [
                "Immediate medical evaluation required",
                "Go to emergency department",
                "Do not delay seeking care"
            ]
            assessment['confidence'] = 0.95
            return assessment
        
        # Calculate symptom acuity scores
        acuity_scores = [symptom.calculate_acuity_score() for symptom in symptoms]
        max_acuity = max(acuity_scores) if acuity_scores else 0
        
        # Assess triggers and patterns
        trigger_analysis = self._analyze_triggers(patient_data.get('triggers', []), symptoms)
        
        # Pattern recognition for functional vs organic
        pattern_assessment = self._assess_symptom_patterns(symptoms, patient_data)
        
        # Determine risk level
        risk_level = self._determine_risk_level(
            max_acuity, 
            pattern_assessment, 
            patient_data.get('vital_signs', {}),
            patient_data.get('medical_history', {})
        )
        assessment['risk_level'] = risk_level
        
        # Generate differential diagnoses
        assessment['differential_diagnoses'] = self._generate_differentials(
            symptoms, pattern_assessment, patient_data
        )
        
        # Create recommendations based on risk level
        assessment['recommended_actions'] = self._generate_recommendations(
            risk_level, symptoms, trigger_analysis
        )
        
        # Add follow-up instructions
        assessment['follow_up'] = self._generate_follow_up(risk_level, symptoms)
        
        # Educational content
        assessment['education'] = self._generate_education(symptoms, trigger_analysis)
        
        # Calculate confidence
        assessment['confidence'] = self._calculate_confidence(
            symptoms, patient_data, pattern_assessment
        )
        
        return assessment
    
    def _parse_symptoms(self, symptom_list: List[Dict]) -> List[Symptom]:
        """Parse raw symptom data into Symptom objects"""
        symptoms = []
        
        for s in symptom_list:
            symptom = Symptom(
                name=s.get('name', ''),
                category=SymptomCategory[s.get('category', 'GASTROINTESTINAL').upper()],
                severity=s.get('severity', 5),
                duration_days=s.get('duration_days', 1),
                onset=s.get('onset', 'gradual'),
                frequency=s.get('frequency', 'intermittent'),
                associated_features=s.get('associated_features', []),
                red_flags=s.get('red_flags', [])
            )
            symptoms.append(symptom)
        
        return symptoms
    
    def _check_red_flags(self, symptoms: List[Symptom], 
                        patient_data: Dict) -> List[str]:
        """Check for presence of red flag symptoms"""
        identified_red_flags = []
        
        # Check symptom-specific red flags
        for symptom in symptoms:
            identified_red_flags.extend(symptom.red_flags)
        
        # Check against red flag database
        for category, flags in self.red_flag_database.items():
            for flag in flags:
                # Simple keyword matching (would be more sophisticated in production)
                for symptom in symptoms:
                    if any(keyword in symptom.name.lower() for keyword in flag.lower().split()):
                        identified_red_flags.append(flag)
        
        # Age-specific red flags
        age = patient_data.get('age', 40)
        if age > 50:
            # New onset symptoms after 50 require more caution
            for symptom in symptoms:
                if symptom.duration_days < 30 and symptom.severity > 6:
                    identified_red_flags.append(f"New severe symptoms at age {age}")
        
        return list(set(identified_red_flags))  # Remove duplicates
    
    def _analyze_triggers(self, triggers: List[Dict], 
                         symptoms: List[Symptom]) -> Dict[str, Any]:
        """Analyze relationship between triggers and symptoms"""
        analysis = {
            'identified_triggers': [],
            'trigger_symptom_correlations': {},
            'pattern': None
        }
        
        for trigger_data in triggers:
            trigger = Trigger(
                type=trigger_data.get('type', 'unknown'),
                name=trigger_data.get('name', ''),
                timing=trigger_data.get('timing', 'hours'),
                strength=trigger_data.get('strength', 0.5),
                evidence_level=trigger_data.get('evidence_level', 'weak')
            )
            
            # Assess trigger-symptom correlation
            if trigger.type == 'dietary':
                # Food triggers often cause symptoms within hours
                if trigger.timing == 'immediate' or trigger.timing == 'hours':
                    trigger.strength = min(1.0, trigger.strength * 1.2)
            elif trigger.type == 'stress':
                # Stress can cause immediate or delayed symptoms
                trigger.strength = min(1.0, trigger.strength * 1.1)
            
            analysis['identified_triggers'].append(trigger)
        
        # Determine pattern
        if len([t for t in analysis['identified_triggers'] if t.type == 'dietary']) > 2:
            analysis['pattern'] = 'food_sensitive'
        elif len([t for t in analysis['identified_triggers'] if t.type == 'stress']) > 1:
            analysis['pattern'] = 'stress_reactive'
        
        return analysis
    
    def _assess_symptom_patterns(self, symptoms: List[Symptom], 
                                patient_data: Dict) -> Dict[str, Any]:
        """Assess patterns suggesting functional vs organic disease"""
        pattern = {
            'functional_features': 0,
            'organic_features': 0,
            'chronicity': 'acute',
            'likely_category': None
        }
        
        # Functional disorder features
        functional_indicators = [
            'symptom variability',
            'stress relationship',
            'no nocturnal symptoms',
            'no weight loss',
            'normal labs'
        ]
        
        # Organic disorder features
        organic_indicators = [
            'progressive symptoms',
            'nocturnal symptoms',
            'weight loss',
            'fever',
            'blood in stool',
            'abnormal labs'
        ]
        
        # Assess duration
        avg_duration = np.mean([s.duration_days for s in symptoms]) if symptoms else 0
        if avg_duration > 90:
            pattern['chronicity'] = 'chronic'
        elif avg_duration > 14:
            pattern['chronicity'] = 'subacute'
        else:
            pattern['chronicity'] = 'acute'
        
        # Count features
        for symptom in symptoms:
            if symptom.frequency == 'intermittent':
                pattern['functional_features'] += 1
            if symptom.severity > 8:
                pattern['organic_features'] += 1
            if 'blood' in symptom.name.lower():
                pattern['organic_features'] += 2
        
        # Determine likely category
        if pattern['organic_features'] > pattern['functional_features']:
            pattern['likely_category'] = 'organic'
        else:
            pattern['likely_category'] = 'functional'
        
        return pattern
    
    def _determine_risk_level(self, max_acuity: float, 
                             pattern_assessment: Dict,
                             vital_signs: Dict,
                             medical_history: Dict) -> RiskLevel:
        """Determine overall risk level based on multiple factors"""
        
        # Check vital signs first
        if self._check_critical_vitals(vital_signs):
            return RiskLevel.CRITICAL
        
        # Acuity-based assessment
        if max_acuity > 30:
            return RiskLevel.VERY_HIGH
        elif max_acuity > 20:
            return RiskLevel.HIGH
        elif max_acuity > 10:
            return RiskLevel.MEDIUM
        
        # Pattern-based modifiers
        if pattern_assessment['likely_category'] == 'organic':
            if pattern_assessment['chronicity'] == 'acute':
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM
        
        # Functional patterns generally lower risk
        if pattern_assessment['likely_category'] == 'functional':
            if pattern_assessment['chronicity'] == 'chronic':
                return RiskLevel.LOW
            else:
                return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _check_critical_vitals(self, vital_signs: Dict) -> bool:
        """Check if vital signs indicate critical condition"""
        critical = False
        
        hr = vital_signs.get('heart_rate', 75)
        if hr > 140 or hr < 40:
            critical = True
        
        sbp = vital_signs.get('systolic_bp', 120)
        if sbp < 90 or sbp > 200:
            critical = True
        
        rr = vital_signs.get('respiratory_rate', 16)
        if rr > 30 or rr < 8:
            critical = True
        
        temp = vital_signs.get('temperature', 98.6)
        if temp > 104 or temp < 95:
            critical = True
        
        return critical
    
    def _generate_differentials(self, symptoms: List[Symptom], 
                               pattern: Dict, 
                               patient_data: Dict) -> List[Dict]:
        """Generate differential diagnoses based on presentation"""
        differentials = []
        
        # Get primary symptom category
        if symptoms:
            primary_symptom = max(symptoms, key=lambda s: s.severity)
            
            if 'pain' in primary_symptom.name.lower():
                if pattern['chronicity'] == 'acute':
                    differentials = self.differential_diagnoses.get('acute_abdominal_pain', [])
                else:
                    differentials = self.differential_diagnoses.get('chronic_symptoms', [])
        
        # Filter based on patient factors
        age = patient_data.get('age', 40)
        if age < 30:
            # Less likely to have serious pathology
            differentials = [d for d in differentials if d['urgency'] != 'critical']
        
        return differentials[:5]  # Return top 5 most likely
    
    def _generate_recommendations(self, risk_level: RiskLevel, 
                                 symptoms: List[Symptom],
                                 trigger_analysis: Dict) -> List[str]:
        """Generate specific recommendations based on assessment"""
        recommendations = []
        
        # Risk level-based primary recommendation
        recommendations.append(risk_level.action_timeframe)
        
        # Symptom-specific recommendations
        if risk_level == RiskLevel.LOW:
            recommendations.extend([
                "Keep symptom diary to track patterns",
                "Stay hydrated",
                "Consider dietary modifications (low FODMAP if IBS suspected)",
                "Stress management techniques",
                "Over-the-counter remedies as appropriate"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Schedule appointment with primary care provider",
                "Prepare list of symptoms and triggers for appointment",
                "Continue symptom diary",
                "Avoid known triggers",
                "Consider basic lab work (CBC, CMP, inflammatory markers)"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Seek urgent medical evaluation",
                "Do not eat or drink if surgery might be needed",
                "Bring list of medications",
                "Have someone drive you to appointment",
                "Bring previous medical records if available"
            ])
        elif risk_level in [RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]:
            recommendations.extend([
                "Go to emergency department immediately",
                "Call 911 if unable to transport safely",
                "Do not drive yourself",
                "Bring medications and medical history",
                "Have emergency contact available"
            ])
        
        # Trigger-specific recommendations
        if trigger_analysis['pattern'] == 'food_sensitive':
            recommendations.append("Consider elimination diet with professional guidance")
        elif trigger_analysis['pattern'] == 'stress_reactive':
            recommendations.append("Consider psychological support or stress management program")
        
        return recommendations
    
    def _generate_follow_up(self, risk_level: RiskLevel, 
                           symptoms: List[Symptom]) -> List[str]:
        """Generate follow-up instructions"""
        follow_up = []
        
        if risk_level == RiskLevel.LOW:
            follow_up = [
                "Return if symptoms worsen or new symptoms develop",
                "Follow up with PCP if no improvement in 2 weeks",
                "Track symptoms for pattern identification"
            ]
        elif risk_level == RiskLevel.MEDIUM:
            follow_up = [
                "Follow up as scheduled with provider",
                "Return sooner if symptoms worsen",
                "Complete recommended testing",
                "Track response to interventions"
            ]
        else:
            follow_up = [
                "Follow discharge instructions carefully",
                "Take medications as prescribed",
                "Return to ED if symptoms recur or worsen",
                "Follow up with specialist as recommended"
            ]
        
        return follow_up
    
    def _generate_education(self, symptoms: List[Symptom], 
                          trigger_analysis: Dict) -> List[str]:
        """Generate educational content"""
        education = []
        
        # General gut-brain education
        education.append("The gut-brain axis involves bidirectional communication between digestive and nervous systems")
        
        # Symptom-specific education
        gi_symptoms = [s for s in symptoms if s.category == SymptomCategory.GASTROINTESTINAL]
        if gi_symptoms:
            education.append("GI symptoms can be influenced by stress, diet, sleep, and gut microbiome")
        
        psych_symptoms = [s for s in symptoms if s.category == SymptomCategory.PSYCHOLOGICAL]
        if psych_symptoms:
            education.append("Mood symptoms can be related to gut health through neurotransmitter production and inflammation")
        
        # Trigger education
        if trigger_analysis['identified_triggers']:
            education.append("Identifying and managing triggers is key to symptom control")
        
        return education
    
    def _calculate_confidence(self, symptoms: List[Symptom], 
                            patient_data: Dict,
                            pattern: Dict) -> float:
        """Calculate confidence in assessment"""
        confidence = 0.5  # Base confidence
        
        # More symptoms increase confidence
        if len(symptoms) > 3:
            confidence += 0.1
        
        # Clear pattern increases confidence
        if pattern['likely_category']:
            confidence += 0.15
        
        # Complete vital signs increase confidence
        if patient_data.get('vital_signs'):
            confidence += 0.1
        
        # Medical history increases confidence
        if patient_data.get('medical_history'):
            confidence += 0.1
        
        # Red flags increase confidence (clear action needed)
        if any(s.red_flags for s in symptoms):
            confidence += 0.2
        
        return min(0.95, confidence)  # Cap at 95%

class SymptomTriggerMatrix:
    """
    Matrix mapping symptoms to triggers with evidence levels
    """
    
    def __init__(self):
        self.matrix = self._build_matrix()
        
    def _build_matrix(self) -> pd.DataFrame:
        """
        Build evidence-based symptom-trigger matrix
        Based on systematic reviews and clinical guidelines
        """
        data = {
            'symptom': [],
            'trigger': [],
            'correlation_strength': [],
            'evidence_level': [],
            'mechanism': [],
            'reference': []
        }
        
        # Dietary triggers (Bohn et al., 2015, Gut)
        dietary_correlations = [
            ('bloating', 'FODMAPs', 0.7, 'strong', 'fermentation', 'Gibson & Shepherd, 2010'),
            ('diarrhea', 'lactose', 0.8, 'strong', 'malabsorption', 'Misselwitz et al., 2019'),
            ('abdominal_pain', 'gluten', 0.5, 'moderate', 'immune/permeability', 'Catassi et al., 2017'),
            ('nausea', 'fatty_foods', 0.6, 'moderate', 'delayed gastric emptying', 'Feinle-Bisset, 2016')
        ]
        
        # Stress triggers (Qin et al., 2014, World J Gastroenterol)
        stress_correlations = [
            ('IBS_symptoms', 'acute_stress', 0.8, 'strong', 'HPA activation', 'Chang, 2011'),
            ('functional_dyspepsia', 'chronic_stress', 0.7, 'strong', 'vagal dysfunction', 'Van Oudenhove et al., 2016'),
            ('abdominal_pain', 'anxiety', 0.6, 'moderate', 'visceral hypersensitivity', 'Greenwood-Van Meerveld, 2017')
        ]
        
        # Compile all correlations
        all_correlations = dietary_correlations + stress_correlations
        
        for correlation in all_correlations:
            data['symptom'].append(correlation[0])
            data['trigger'].append(correlation[1])
            data['correlation_strength'].append(correlation[2])
            data['evidence_level'].append(correlation[3])
            data['mechanism'].append(correlation[4])
            data['reference'].append(correlation[5])
        
        return pd.DataFrame(data)
    
    def get_trigger_probability(self, symptom: str, trigger: str) -> float:
        """Get probability that a trigger causes a symptom"""
        result = self.matrix[
            (self.matrix['symptom'] == symptom) & 
            (self.matrix['trigger'] == trigger)
        ]
        
        if not result.empty:
            return result.iloc[0]['correlation_strength']
        return 0.0
    
    def get_likely_triggers(self, symptom: str, threshold: float = 0.5) -> List[Dict]:
        """Get likely triggers for a symptom above threshold"""
        results = self.matrix[
            (self.matrix['symptom'] == symptom) & 
            (self.matrix['correlation_strength'] >= threshold)
        ]
        
        return results.to_dict('records')

class ClinicalDecisionSupport:
    """
    Main interface for clinical decision support system
    """
    
    def __init__(self):
        self.risk_assessor = ClinicalRiskAssessment()
        self.symptom_trigger_matrix = SymptomTriggerMatrix()
        self.patient_history = {}
        
    def evaluate_patient(self, patient_input: Dict) -> Dict[str, Any]:
        """
        Main entry point for patient evaluation
        Returns comprehensive assessment and recommendations
        """
        
        # Validate input
        if not self._validate_input(patient_input):
            return {
                'error': 'Invalid input data',
                'message': 'Please provide required patient information'
            }
        
        # Perform risk assessment
        assessment = self.risk_assessor.assess_patient(patient_input)
        
        # Add trigger correlations
        symptoms = patient_input.get('symptoms', [])
        for symptom in symptoms:
            symptom_name = symptom.get('name', '')
            likely_triggers = self.symptom_trigger_matrix.get_likely_triggers(symptom_name)
            symptom['likely_triggers'] = likely_triggers
        
        # Create visualization data
        visualization = self._create_visualization_data(assessment, patient_input)
        
        # Store in history if patient ID provided
        if patient_input.get('patient_id'):
            self._update_patient_history(patient_input['patient_id'], assessment)
        
        # Compile final output
        output = {
            'assessment': assessment,
            'visualization': visualization,
            'symptom_trigger_analysis': symptoms,
            'disclaimer': "This assessment is for clinical decision support only. "
                        "It does not replace professional medical judgment. "
                        "Always consult qualified healthcare providers for medical decisions."
        }
        
        return output
    
    def _validate_input(self, patient_input: Dict) -> bool:
        """Validate that required input fields are present"""
        required_fields = ['symptoms', 'age']
        return all(field in patient_input for field in required_fields)
    
    def _create_visualization_data(self, assessment: Dict, 
                                  patient_input: Dict) -> Dict:
        """Create data structure for visualization"""
        return {
            'risk_gauge': {
                'level': assessment['risk_level'].name if assessment['risk_level'] else 'UNKNOWN',
                'color': assessment['risk_level'].color_code if assessment['risk_level'] else 'gray',
                'confidence': assessment['confidence']
            },
            'symptom_timeline': self._create_symptom_timeline(patient_input.get('symptoms', [])),
            'trigger_correlation_map': self._create_trigger_map(patient_input),
            'action_priority': assessment['recommended_actions'][:3] if assessment['recommended_actions'] else []
        }
    
    def _create_symptom_timeline(self, symptoms: List[Dict]) -> List[Dict]:
        """Create timeline visualization data"""
        timeline = []
        for symptom in symptoms:
            timeline.append({
                'name': symptom.get('name'),
                'start': -symptom.get('duration_days', 0),
                'severity': symptom.get('severity', 5),
                'category': symptom.get('category', 'unknown')
            })
        return sorted(timeline, key=lambda x: x['start'])
    
    def _create_trigger_map(self, patient_input: Dict) -> Dict:
        """Create trigger correlation visualization"""
        triggers = patient_input.get('triggers', [])
        symptoms = patient_input.get('symptoms', [])
        
        connections = []
        for trigger in triggers:
            for symptom in symptoms:
                correlation = self.symptom_trigger_matrix.get_trigger_probability(
                    symptom.get('name', ''),
                    trigger.get('name', '')
                )
                if correlation > 0:
                    connections.append({
                        'source': trigger.get('name'),
                        'target': symptom.get('name'),
                        'strength': correlation
                    })
        
        return {
            'nodes': [
                {'id': t.get('name'), 'type': 'trigger'} for t in triggers
            ] + [
                {'id': s.get('name'), 'type': 'symptom'} for s in symptoms
            ],
            'links': connections
        }
    
    def _update_patient_history(self, patient_id: str, assessment: Dict):
        """Update patient history for longitudinal tracking"""
        if patient_id not in self.patient_history:
            self.patient_history[patient_id] = []
        
        self.patient_history[patient_id].append({
            'timestamp': assessment['timestamp'],
            'risk_level': assessment['risk_level'].name if assessment['risk_level'] else 'UNKNOWN',
            'primary_concerns': assessment['primary_concerns']
        })

def demonstrate_clinical_system():
    """
    Demonstrate the clinical decision support system with example cases
    """
    print("GUT-BRAIN CLINICAL DECISION SUPPORT SYSTEM")
    print("-" * 60)
    
    # Initialize system
    system = ClinicalDecisionSupport()
    
    # Test Case 1: Low Risk - Functional symptoms
    print("\nCASE 1: Functional GI Symptoms")
    print("-" * 40)
    
    case1 = {
        'patient_id': 'PT001',
        'age': 28,
        'gender': 'female',
        'symptoms': [
            {
                'name': 'bloating',
                'category': 'gastrointestinal',
                'severity': 6,
                'duration_days': 180,
                'onset': 'gradual',
                'frequency': 'intermittent'
            },
            {
                'name': 'abdominal discomfort',
                'category': 'gastrointestinal', 
                'severity': 4,
                'duration_days': 180,
                'onset': 'gradual',
                'frequency': 'intermittent'
            }
        ],
        'triggers': [
            {'type': 'dietary', 'name': 'FODMAPs', 'timing': 'hours'},
            {'type': 'stress', 'name': 'work_stress', 'timing': 'days'}
        ],
        'vital_signs': {
            'heart_rate': 72,
            'blood_pressure': '118/76',
            'temperature': 98.6
        }
    }
    
    result1 = system.evaluate_patient(case1)
    print(f"Risk Level: {result1['assessment']['risk_level'].name}")
    print(f"Confidence: {result1['assessment']['confidence']:.1%}")
    print(f"Primary Recommendation: {result1['assessment']['recommended_actions'][0]}")
    
    # Test Case 2: High Risk - Acute symptoms with red flags
    print("\nCASE 2: Acute Abdomen with Red Flags")
    print("-" * 40)
    
    case2 = {
        'patient_id': 'PT002',
        'age': 55,
        'gender': 'male',
        'symptoms': [
            {
                'name': 'severe abdominal pain',
                'category': 'gastrointestinal',
                'severity': 9,
                'duration_days': 0.5,
                'onset': 'sudden',
                'frequency': 'constant',
                'red_flags': ['rigid abdomen']
            },
            {
                'name': 'fever',
                'category': 'systemic',
                'severity': 7,
                'duration_days': 0.5,
                'onset': 'sudden'
            }
        ],
        'vital_signs': {
            'heart_rate': 110,
            'systolic_bp': 95,
            'temperature': 102.5
        }
    }
    
    result2 = system.evaluate_patient(case2)
    print(f"Risk Level: {result2['assessment']['risk_level'].name}")
    print(f"Confidence: {result2['assessment']['confidence']:.1%}")
    print(f"Primary Concerns: {', '.join(result2['assessment']['primary_concerns'][:3])}")
    print(f"Primary Recommendation: {result2['assessment']['recommended_actions'][0]}")
    
    # Test Case 3: Medium Risk - Chronic with new features
    print("\nCASE 3: Chronic Symptoms with New Features")
    print("-" * 40)
    
    case3 = {
        'patient_id': 'PT003',
        'age': 42,
        'gender': 'female',
        'symptoms': [
            {
                'name': 'alternating bowel habits',
                'category': 'gastrointestinal',
                'severity': 5,
                'duration_days': 365,
                'onset': 'gradual',
                'frequency': 'intermittent'
            },
            {
                'name': 'unintentional weight loss',
                'category': 'systemic',
                'severity': 6,
                'duration_days': 60,
                'onset': 'gradual',
                'red_flags': ['weight loss >10%']
            }
        ],
        'medical_history': {
            'family_history': ['colon_cancer'],
            'medications': ['PPIs']
        }
    }
    
    result3 = system.evaluate_patient(case3)
    print(f"Risk Level: {result3['assessment']['risk_level'].name}")
    print(f"Confidence: {result3['assessment']['confidence']:.1%}")
    print(f"Differential Diagnoses: {len(result3['assessment']['differential_diagnoses'])}")
    print(f"Primary Recommendation: {result3['assessment']['recommended_actions'][0]}")
    
    print("\n" + "-" * 60)
    print("DEMONSTRATION COMPLETE")
    print("\nDISCLAIMER: This system is for clinical decision support only.")
    print("Always consult qualified healthcare providers for medical decisions.")

if __name__ == "__main__":
    demonstrate_clinical_system()

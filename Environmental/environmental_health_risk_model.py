"""
ENVIRONMENTAL HEALTH RISK ASSESSMENT MODEL
Based on EPA Risk Assessment Framework and Environmental Health Research

This model integrates environmental exposures with health outcomes following:
- EPA Risk Assessment Guidelines (EPA/630/P-03/001F)
- WHO Environmental Health Criteria
- NIH Environmental Health Sciences Framework

Key Components:
1. Hazard Identification - environmental stressors affecting health
2. Dose-Response Assessment - exposure levels and health effects
3. Exposure Assessment - individual and population exposure patterns
4. Risk Characterization - integrated risk scoring and recommendations

References:
- Bell et al., 2021, Environmental Research - Air pollution and health
- Lucas et al., 2018, Photochemistry & Photobiology - UV and Vitamin D
- D'Amato et al., 2020, Allergy - Climate change and allergic diseases
- Klepeis et al., 2001, J Exposure Science - Indoor/outdoor time patterns

Version: 1.0.0
Author: Cazzy Aporbo updated 10/1/2025
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum, auto
from datetime import datetime, timedelta, time
import json
import logging
from collections import defaultdict
import warnings
import math
from scipy.interpolate import interp1d
from scipy.stats import norm, lognorm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthRiskLevel(Enum):
    """EPA-based risk levels with health action thresholds"""
    MINIMAL = auto()  # No concern
    LOW = auto()  # Acceptable risk
    MODERATE = auto()  # Some concern, preventive measures recommended
    HIGH = auto()  # Significant concern, protective actions needed
    SEVERE = auto()  # Hazardous, avoid exposure
    
    @property
    def color_code(self) -> str:
        colors = {
            HealthRiskLevel.MINIMAL: "green",
            HealthRiskLevel.LOW: "yellow-green", 
            HealthRiskLevel.MODERATE: "yellow",
            HealthRiskLevel.HIGH: "orange",
            HealthRiskLevel.SEVERE: "red"
        }
        return colors[self]
    
    @property
    def action_threshold(self) -> str:
        actions = {
            HealthRiskLevel.MINIMAL: "No special precautions needed",
            HealthRiskLevel.LOW: "Standard health practices sufficient",
            HealthRiskLevel.MODERATE: "Consider protective measures",
            HealthRiskLevel.HIGH: "Implement protective measures, limit exposure",
            HealthRiskLevel.SEVERE: "Avoid exposure, seek indoor shelter or protection"
        }
        return actions[self]

@dataclass
class EnvironmentalExposures:
    """
    Current environmental exposure levels with health benchmarks
    """
    # Air Quality (EPA AQI standards)
    pm25: float = 12.0  # μg/m³ (EPA annual standard: 12, WHO: 5)
    pm10: float = 20.0  # μg/m³ (EPA 24hr: 150)
    ozone: float = 0.05  # ppm (EPA 8hr: 0.070)
    no2: float = 20.0  # ppb (EPA annual: 53)
    so2: float = 10.0  # ppb (EPA 1hr: 75)
    co: float = 2.0  # ppm (EPA 8hr: 9)
    aqi: int = 50  # Air Quality Index (0-500)
    
    # UV Radiation (EPA UV Index)
    uv_index: float = 5.0  # 0-11+ scale
    solar_radiation: float = 500.0  # W/m²
    
    # Temperature & Humidity
    temperature: float = 72.0  # °F
    humidity: float = 45.0  # %
    heat_index: float = 75.0  # °F
    wind_chill: float = 70.0  # °F
    
    # Biological (from monitoring stations)
    pollen_count: int = 500  # grains/m³
    mold_spores: int = 1000  # spores/m³
    
    # Indoor Environment
    indoor_temp: float = 68.0  # °F
    indoor_humidity: float = 45.0  # %
    indoor_co2: float = 800.0  # ppm
    indoor_voc: float = 0.3  # mg/m³
    radon: float = 1.0  # pCi/L
    
    # Water Quality (if applicable)
    water_ph: float = 7.0
    water_tds: float = 200.0  # mg/L
    
    # Noise
    noise_level: float = 55.0  # dBA
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_aqi_from_pm25(self) -> int:
        """
        Calculate AQI from PM2.5 using EPA formula
        EPA Technical Assistance Document, 2016
        """
        # EPA AQI breakpoints for PM2.5 (μg/m³)
        breakpoints = [
            (0, 12.0, 0, 50),  # Good
            (12.1, 35.4, 51, 100),  # Moderate
            (35.5, 55.4, 101, 150),  # Unhealthy for sensitive
            (55.5, 150.4, 151, 200),  # Unhealthy
            (150.5, 250.4, 201, 300),  # Very unhealthy
            (250.5, 350.4, 301, 400),  # Hazardous
            (350.5, 500.4, 401, 500)  # Hazardous
        ]
        
        for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
            if bp_lo <= self.pm25 <= bp_hi:
                # Linear interpolation formula
                aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (self.pm25 - bp_lo) + aqi_lo
                return int(aqi)
        
        return 500 if self.pm25 > 500.4 else 0

@dataclass 
class PersonalFactors:
    """Individual susceptibility factors affecting environmental health risk"""
    
    age: int = 30
    gender: str = "female"
    
    # Health conditions increasing vulnerability
    asthma: bool = False
    cardiovascular_disease: bool = False
    diabetes: bool = False
    pregnancy: bool = False
    immunocompromised: bool = False
    
    # Allergies
    pollen_allergy: bool = False
    dust_allergy: bool = False
    mold_allergy: bool = False
    
    # Lifestyle factors
    outdoor_activity_hours: float = 2.0  # hours/day
    exercise_outdoors: bool = True
    occupation_outdoor: bool = False
    
    # Protective measures
    uses_air_purifier: bool = False
    takes_allergy_medication: bool = False
    uses_sunscreen_regularly: bool = True
    
    # Skin type for UV sensitivity (Fitzpatrick scale)
    skin_type: int = 3  # 1-6, lower = more sensitive
    
    def calculate_vulnerability_score(self) -> float:
        """
        Calculate personal vulnerability to environmental exposures
        Based on EPA Environmental Justice screening
        """
        score = 1.0  # Baseline
        
        # Age factors (Sacks et al., 2011, Environ Health Perspect)
        if self.age < 5 or self.age > 65:
            score *= 1.5
        elif self.age < 18:
            score *= 1.2
            
        # Health conditions (EPA, 2019, Integrated Science Assessments)
        if self.asthma:
            score *= 1.4
        if self.cardiovascular_disease:
            score *= 1.3
        if self.diabetes:
            score *= 1.2
        if self.pregnancy:
            score *= 1.3
        if self.immunocompromised:
            score *= 1.5
            
        # Exposure duration
        if self.outdoor_activity_hours > 4:
            score *= 1.2
        if self.occupation_outdoor:
            score *= 1.3
            
        # Protective factors reduce vulnerability
        if self.uses_air_purifier:
            score *= 0.9
        if self.takes_allergy_medication and (self.pollen_allergy or self.dust_allergy):
            score *= 0.85
            
        return min(3.0, score)  # Cap at 3x baseline

class AirQualityAssessment:
    """
    Air quality health impact assessment based on EPA and WHO guidelines
    """
    
    def __init__(self):
        self.health_thresholds = self._initialize_thresholds()
        self.dose_response_curves = self._initialize_dose_response()
        
    def _initialize_thresholds(self) -> Dict[str, Dict]:
        """
        Health-based thresholds from EPA NAAQS and WHO guidelines
        """
        return {
            'pm25': {
                'who_annual': 5,  # WHO 2021 guideline
                'who_24hr': 15,
                'epa_annual': 12,  # EPA NAAQS
                'epa_24hr': 35,
                'sensitive_groups': 9  # More protective for vulnerable
            },
            'ozone': {
                'who_8hr': 0.05,  # ppm
                'epa_8hr': 0.070,
                'sensitive_groups': 0.060
            },
            'no2': {
                'who_annual': 10,  # ppb  
                'epa_annual': 53,
                'sensitive_groups': 30
            }
        }
    
    def _initialize_dose_response(self) -> Dict[str, callable]:
        """
        Dose-response functions from epidemiological studies
        """
        # PM2.5 relative risk per 10 μg/m³ increase
        # Burnett et al., 2018, PNAS - Global Exposure Mortality Model
        def pm25_mortality_risk(concentration: float) -> float:
            if concentration <= 2.4:
                return 1.0
            # Supralinear at low concentrations, sublinear at high
            rr = 1.0 + 0.08 * np.log(concentration / 2.4)
            return min(2.0, rr)
        
        # Respiratory symptoms (Orellano et al., 2020, Environ Res)
        def pm25_respiratory_risk(concentration: float) -> float:
            # 3.2% increase per 10 μg/m³
            return 1.0 + 0.032 * (concentration / 10)
        
        # Cardiovascular (Rajagopalan et al., 2018, JACC)
        def pm25_cardiovascular_risk(concentration: float) -> float:
            # 1% increase per 10 μg/m³
            return 1.0 + 0.01 * (concentration / 10)
        
        return {
            'pm25_mortality': pm25_mortality_risk,
            'pm25_respiratory': pm25_respiratory_risk,
            'pm25_cardiovascular': pm25_cardiovascular_risk
        }
    
    def assess_health_impact(self, exposures: EnvironmentalExposures,
                           personal: PersonalFactors) -> Dict[str, Any]:
        """
        Assess health impacts from air quality exposure
        """
        impacts = {
            'aqi_category': self._get_aqi_category(exposures.aqi),
            'health_effects': [],
            'inflammation_increase': 0.0,
            'respiratory_risk': 1.0,
            'cardiovascular_risk': 1.0,
            'recommendations': []
        }
        
        # Calculate inflammation increase
        # Brook et al., 2010, Circulation - PM2.5 and inflammation
        if exposures.pm25 > 12:
            impacts['inflammation_increase'] = min(20, (exposures.pm25 - 12) * 0.5)
        
        # Apply dose-response functions
        impacts['respiratory_risk'] = self.dose_response_curves['pm25_respiratory'](
            exposures.pm25
        )
        impacts['cardiovascular_risk'] = self.dose_response_curves['pm25_cardiovascular'](
            exposures.pm25
        )
        
        # Adjust for personal vulnerability
        vulnerability = personal.calculate_vulnerability_score()
        impacts['respiratory_risk'] *= vulnerability
        impacts['cardiovascular_risk'] *= vulnerability
        
        # Generate health effects based on AQI
        if exposures.aqi > 100:
            impacts['health_effects'].append("Respiratory irritation possible")
        if exposures.aqi > 150:
            impacts['health_effects'].append("Increased respiratory symptoms")
            impacts['health_effects'].append("Aggravation of heart/lung disease")
        if exposures.aqi > 200:
            impacts['health_effects'].append("Significant health effects")
            
        # Recommendations based on conditions
        impacts['recommendations'] = self._generate_air_quality_recommendations(
            exposures, personal
        )
        
        return impacts
    
    def _get_aqi_category(self, aqi: int) -> str:
        """EPA AQI categories"""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    def _generate_air_quality_recommendations(self, exposures: EnvironmentalExposures,
                                             personal: PersonalFactors) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if exposures.aqi > 100:
            recommendations.append("Limit prolonged outdoor exertion")
            if personal.asthma:
                recommendations.append("Keep quick-relief medicine handy")
                recommendations.append("Monitor symptoms closely")
        
        if exposures.aqi > 150:
            recommendations.append("Avoid outdoor exercise")
            recommendations.append("Keep windows closed")
            recommendations.append("Use air purifier if available")
            
        if exposures.pm25 > 35:
            recommendations.append("Consider wearing N95 mask outdoors")
            
        if personal.outdoor_activity_hours > 2 and exposures.aqi > 50:
            recommendations.append(f"Reduce outdoor time to <{personal.outdoor_activity_hours/2:.1f} hours")
            
        return recommendations

class UVRadiationAssessment:
    """
    UV radiation exposure and Vitamin D synthesis assessment
    """
    
    def __init__(self):
        self.vitamin_d_model = self._initialize_vitamin_d_model()
        
    def _initialize_vitamin_d_model(self) -> Dict:
        """
        Vitamin D synthesis model based on Webb et al., 2018, Photochem Photobiol
        """
        return {
            'synthesis_threshold': 3.0,  # UV index needed for synthesis
            'optimal_exposure_minutes': {
                1: 60,  # Skin type 1 (very fair)
                2: 45,
                3: 30,
                4: 20,
                5: 15,
                6: 10   # Skin type 6 (very dark)
            },
            'daily_requirement_iu': {
                'female': 600,  # 19-50 years
                'female_pregnant': 600,
                'female_elderly': 800,  # >70 years
                'male': 600,
                'male_elderly': 800
            }
        }
    
    def assess_uv_exposure(self, exposures: EnvironmentalExposures,
                          personal: PersonalFactors,
                          time_of_day: int = 12) -> Dict[str, Any]:
        """
        Assess UV exposure risks and Vitamin D synthesis
        Based on EPA SunWise program and Holick, 2007, NEJM
        """
        assessment = {
            'uv_risk_level': self._get_uv_risk_level(exposures.uv_index),
            'burn_time_minutes': 0,
            'vitamin_d_synthesis': {},
            'protection_needed': False,
            'recommendations': []
        }
        
        # Calculate burn time (Fitzpatrick & Sober, 1985, Arch Dermatol)
        base_burn_time = {
            1: 10,  # Very fair skin
            2: 15,
            3: 20,
            4: 30,
            5: 45,
            6: 60   # Very dark skin
        }
        
        if exposures.uv_index > 0:
            assessment['burn_time_minutes'] = base_burn_time.get(
                personal.skin_type, 20
            ) * (3 / exposures.uv_index)
        
        # Vitamin D synthesis calculation
        # Engelsen, 2010, Nutrients - The radiation model
        if exposures.uv_index >= self.vitamin_d_model['synthesis_threshold']:
            optimal_minutes = self.vitamin_d_model['optimal_exposure_minutes'].get(
                personal.skin_type, 20
            )
            
            # Adjust for time of day (peak synthesis 10am-3pm)
            time_factor = 1.0
            if time_of_day < 10 or time_of_day > 15:
                time_factor = 0.5
            
            # Calculate synthesis potential
            synthesis_rate = (exposures.uv_index / 10) * time_factor
            
            # Account for skin exposure (McKenzie et al., 2009, Photochem Photobiol)
            exposed_skin_fraction = 0.25  # Arms and face typically
            
            assessment['vitamin_d_synthesis'] = {
                'optimal_exposure_minutes': optimal_minutes,
                'synthesis_rate_iu_per_min': synthesis_rate * 100 * exposed_skin_fraction,
                'daily_requirement_iu': self._get_vitamin_d_requirement(personal),
                'sufficient': optimal_minutes * synthesis_rate * 100 * exposed_skin_fraction >= 400
            }
        else:
            assessment['vitamin_d_synthesis'] = {
                'optimal_exposure_minutes': 0,
                'synthesis_rate_iu_per_min': 0,
                'daily_requirement_iu': self._get_vitamin_d_requirement(personal),
                'sufficient': False
            }
        
        # Protection recommendations
        if exposures.uv_index >= 3:
            assessment['protection_needed'] = True
            
        if exposures.uv_index >= 6:
            assessment['recommendations'].extend([
                "Wear SPF 30+ sunscreen",
                "Seek shade during midday hours",
                "Wear protective clothing and hat",
                "Use UV-blocking sunglasses"
            ])
        elif exposures.uv_index >= 3:
            assessment['recommendations'].extend([
                "Use SPF 15+ sunscreen for extended exposure",
                "Wear sunglasses"
            ])
            
        # Vitamin D recommendations
        if not assessment['vitamin_d_synthesis']['sufficient']:
            assessment['recommendations'].append(
                f"Consider Vitamin D supplementation ({self._get_vitamin_d_requirement(personal)} IU/day)"
            )
        else:
            assessment['recommendations'].append(
                f"{assessment['vitamin_d_synthesis']['optimal_exposure_minutes']:.0f} minutes "
                "unprotected sun exposure sufficient for Vitamin D"
            )
            
        return assessment
    
    def _get_uv_risk_level(self, uv_index: float) -> HealthRiskLevel:
        """EPA UV Index risk levels"""
        if uv_index < 3:
            return HealthRiskLevel.LOW
        elif uv_index < 6:
            return HealthRiskLevel.MODERATE
        elif uv_index < 8:
            return HealthRiskLevel.HIGH
        elif uv_index < 11:
            return HealthRiskLevel.SEVERE
        else:
            return HealthRiskLevel.SEVERE
    
    def _get_vitamin_d_requirement(self, personal: PersonalFactors) -> int:
        """Get daily Vitamin D requirement based on demographics"""
        if personal.gender == "female":
            if personal.pregnancy:
                return self.vitamin_d_model['daily_requirement_iu']['female_pregnant']
            elif personal.age >= 70:
                return self.vitamin_d_model['daily_requirement_iu']['female_elderly']
            else:
                return self.vitamin_d_model['daily_requirement_iu']['female']
        else:
            if personal.age >= 70:
                return self.vitamin_d_model['daily_requirement_iu']['male_elderly']
            else:
                return self.vitamin_d_model['daily_requirement_iu']['male']

class PollenAllergenAssessment:
    """
    Pollen and allergen exposure assessment
    """
    
    def __init__(self):
        self.pollen_thresholds = self._initialize_pollen_thresholds()
        
    def _initialize_pollen_thresholds(self) -> Dict[str, Dict]:
        """
        Pollen count thresholds from AAAAI and NAB
        """
        return {
            'tree': {
                'low': (1, 14),
                'moderate': (15, 89),
                'high': (90, 1499),
                'very_high': (1500, float('inf'))
            },
            'grass': {
                'low': (1, 4),
                'moderate': (5, 19),
                'high': (20, 199),
                'very_high': (200, float('inf'))
            },
            'weed': {
                'low': (1, 9),
                'moderate': (10, 49),
                'high': (50, 499),
                'very_high': (500, float('inf'))
            }
        }
    
    def assess_allergen_exposure(self, exposures: EnvironmentalExposures,
                                personal: PersonalFactors) -> Dict[str, Any]:
        """
        Assess allergen exposure and health impacts
        D'Amato et al., 2020, Allergy - Climate change and allergic diseases
        """
        assessment = {
            'pollen_level': self._categorize_pollen_level(exposures.pollen_count),
            'symptom_risk': 'low',
            'inflammation_trigger': False,
            'sleep_quality_impact': 0,
            'recommendations': []
        }
        
        # Calculate symptom risk for allergic individuals
        if personal.pollen_allergy:
            if exposures.pollen_count > 50:
                assessment['symptom_risk'] = 'moderate'
                assessment['sleep_quality_impact'] = 10  # % reduction
            if exposures.pollen_count > 500:
                assessment['symptom_risk'] = 'high'
                assessment['inflammation_trigger'] = True
                assessment['sleep_quality_impact'] = 25
                
        # Mold exposure assessment
        if personal.mold_allergy and exposures.mold_spores > 3000:
            assessment['symptom_risk'] = 'high'
            assessment['inflammation_trigger'] = True
            
        # Generate recommendations
        if assessment['symptom_risk'] != 'low':
            assessment['recommendations'].extend([
                "Keep windows closed",
                "Shower and change clothes after outdoor activities",
                "Use HEPA air filter indoors"
            ])
            
            if personal.pollen_allergy and not personal.takes_allergy_medication:
                assessment['recommendations'].append(
                    "Consider starting antihistamines before symptoms begin"
                )
                
            if exposures.pollen_count > 500:
                assessment['recommendations'].append(
                    "Limit outdoor activities, especially early morning"
                )
                
        return assessment
    
    def _categorize_pollen_level(self, count: int) -> str:
        """Categorize pollen count"""
        if count < 50:
            return "Low"
        elif count < 150:
            return "Moderate"
        elif count < 500:
            return "High"
        else:
            return "Very High"

class IndoorEnvironmentAssessment:
    """
    Indoor environmental quality assessment
    Based on EPA Indoor Air Quality guidelines and ASHRAE standards
    """
    
    def __init__(self):
        self.optimal_ranges = self._initialize_optimal_ranges()
        
    def _initialize_optimal_ranges(self) -> Dict[str, Dict]:
        """
        Optimal indoor environment ranges from ASHRAE 55 and EPA
        """
        return {
            'temperature': {
                'sleep': (65, 68),  # °F, Okamoto-Mizuno & Mizuno, 2012, J Physiol Anthropol
                'daytime': (68, 76),  # ASHRAE 55-2020
                'exercise': (64, 68)
            },
            'humidity': {
                'optimal': (40, 50),  # % RH
                'acceptable': (30, 60),
                'mold_risk': 60
            },
            'co2': {
                'excellent': (0, 600),  # ppm
                'good': (600, 800),
                'acceptable': (800, 1000),
                'poor': (1000, 2500),
                'unacceptable': (2500, float('inf'))
            },
            'light': {
                'morning': 10000,  # lux, for circadian rhythm
                'daytime': 500,
                'evening': 100,
                'night': 10
            }
        }
    
    def assess_indoor_environment(self, exposures: EnvironmentalExposures,
                                 time_of_day: int = 12) -> Dict[str, Any]:
        """
        Assess indoor environmental quality
        """
        assessment = {
            'overall_quality': 'good',
            'temperature_status': {},
            'humidity_status': {},
            'air_quality_status': {},
            'light_status': {},
            'health_impacts': [],
            'recommendations': []
        }
        
        # Temperature assessment
        current_range = 'daytime' if 7 <= time_of_day <= 22 else 'sleep'
        optimal = self.optimal_ranges['temperature'][current_range]
        
        if optimal[0] <= exposures.indoor_temp <= optimal[1]:
            assessment['temperature_status']['status'] = 'optimal'
        else:
            assessment['temperature_status']['status'] = 'suboptimal'
            assessment['temperature_status']['adjustment'] = (
                f"Adjust to {optimal[0]}-{optimal[1]}°F"
            )
            assessment['health_impacts'].append("May affect sleep quality" if current_range == 'sleep' else "May affect comfort")
            
        # Humidity assessment
        if self.optimal_ranges['humidity']['optimal'][0] <= exposures.indoor_humidity <= self.optimal_ranges['humidity']['optimal'][1]:
            assessment['humidity_status']['status'] = 'optimal'
        elif exposures.indoor_humidity < 30:
            assessment['humidity_status']['status'] = 'too_dry'
            assessment['health_impacts'].append("Increased respiratory irritation risk")
            assessment['recommendations'].append("Use humidifier")
        elif exposures.indoor_humidity > 60:
            assessment['humidity_status']['status'] = 'too_humid'
            assessment['health_impacts'].append("Increased mold/allergen risk")
            assessment['recommendations'].append("Use dehumidifier or increase ventilation")
            
        # CO2 assessment (cognitive function impact)
        # Satish et al., 2012, Environ Health Perspect - CO2 and decision making
        if exposures.indoor_co2 < 800:
            assessment['air_quality_status']['co2'] = 'excellent'
        elif exposures.indoor_co2 < 1000:
            assessment['air_quality_status']['co2'] = 'good'
        else:
            assessment['air_quality_status']['co2'] = 'poor'
            cognitive_decline = min(15, (exposures.indoor_co2 - 1000) * 0.01)
            assessment['health_impacts'].append(f"Cognitive performance reduced by ~{cognitive_decline:.0f}%")
            assessment['recommendations'].append("Increase ventilation")
            
        # Overall assessment
        if (assessment['temperature_status'].get('status') != 'optimal' or
            assessment['humidity_status'].get('status') != 'optimal' or
            assessment['air_quality_status'].get('co2') not in ['excellent', 'good']):
            assessment['overall_quality'] = 'needs_improvement'
            
        return assessment

class IntegratedEnvironmentalHealthModel:
    """
    Main integrated environmental health risk assessment model
    """
    
    def __init__(self):
        self.air_quality = AirQualityAssessment()
        self.uv_radiation = UVRadiationAssessment()
        self.pollen = PollenAllergenAssessment()
        self.indoor = IndoorEnvironmentAssessment()
        
        # Historical data storage
        self.exposure_history = defaultdict(list)
        self.health_outcomes = defaultdict(list)
        
    def comprehensive_assessment(self, exposures: EnvironmentalExposures,
                                personal: PersonalFactors,
                                location: str = "home") -> Dict[str, Any]:
        """
        Perform comprehensive environmental health assessment
        """
        current_hour = datetime.now().hour
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'location': location,
            'overall_risk': HealthRiskLevel.LOW,
            'risk_components': {},
            'health_impacts': {
                'immediate': [],
                'short_term': [],
                'long_term': []
            },
            'personalized_recommendations': [],
            'protective_actions': [],
            'exposure_windows': {}
        }
        
        # 1. Air Quality Assessment
        air_assessment = self.air_quality.assess_health_impact(exposures, personal)
        results['risk_components']['air_quality'] = {
            'aqi': exposures.aqi,
            'category': air_assessment['aqi_category'],
            'inflammation_increase': air_assessment['inflammation_increase'],
            'risk_level': self._calculate_air_risk_level(exposures.aqi, personal)
        }
        results['health_impacts']['immediate'].extend(air_assessment['health_effects'])
        results['personalized_recommendations'].extend(air_assessment['recommendations'])
        
        # 2. UV Radiation Assessment
        uv_assessment = self.uv_radiation.assess_uv_exposure(exposures, personal, current_hour)
        results['risk_components']['uv_radiation'] = {
            'uv_index': exposures.uv_index,
            'risk_level': uv_assessment['uv_risk_level'],
            'vitamin_d_status': uv_assessment['vitamin_d_synthesis']
        }
        results['personalized_recommendations'].extend(uv_assessment['recommendations'])
        
        # 3. Pollen/Allergen Assessment
        pollen_assessment = self.pollen.assess_allergen_exposure(exposures, personal)
        results['risk_components']['allergens'] = {
            'pollen_level': pollen_assessment['pollen_level'],
            'symptom_risk': pollen_assessment['symptom_risk'],
            'inflammation_trigger': pollen_assessment['inflammation_trigger']
        }
        if pollen_assessment['inflammation_trigger']:
            results['health_impacts']['immediate'].append("Allergic inflammation likely")
        results['personalized_recommendations'].extend(pollen_assessment['recommendations'])
        
        # 4. Indoor Environment Assessment
        indoor_assessment = self.indoor.assess_indoor_environment(exposures, current_hour)
        results['risk_components']['indoor_environment'] = {
            'overall_quality': indoor_assessment['overall_quality'],
            'optimization_needed': indoor_assessment['overall_quality'] != 'good'
        }
        results['health_impacts']['short_term'].extend(indoor_assessment['health_impacts'])
        results['personalized_recommendations'].extend(indoor_assessment['recommendations'])
        
        # 5. Calculate optimal exposure windows
        results['exposure_windows'] = self._calculate_exposure_windows(exposures, personal)
        
        # 6. Determine overall risk level
        results['overall_risk'] = self._determine_overall_risk(results['risk_components'])
        
        # 7. Generate integrated recommendations
        results['protective_actions'] = self._generate_protective_actions(
            results['overall_risk'], 
            results['risk_components'],
            personal
        )
        
        # 8. Calculate cumulative exposure burden
        results['cumulative_burden'] = self._calculate_cumulative_burden(exposures, personal)
        
        return results
    
    def _calculate_air_risk_level(self, aqi: int, personal: PersonalFactors) -> HealthRiskLevel:
        """Calculate air quality risk level with personal factors"""
        
        # Base risk from AQI
        if aqi <= 50:
            base_risk = HealthRiskLevel.MINIMAL
        elif aqi <= 100:
            base_risk = HealthRiskLevel.LOW
        elif aqi <= 150:
            base_risk = HealthRiskLevel.MODERATE
        elif aqi <= 200:
            base_risk = HealthRiskLevel.HIGH
        else:
            base_risk = HealthRiskLevel.SEVERE
            
        # Adjust for vulnerable populations
        if personal.asthma or personal.cardiovascular_disease:
            if aqi > 50 and base_risk != HealthRiskLevel.SEVERE:
                # Increase risk level by one for sensitive individuals
                risk_levels = list(HealthRiskLevel)
                current_index = risk_levels.index(base_risk)
                if current_index < len(risk_levels) - 1:
                    base_risk = risk_levels[current_index + 1]
                    
        return base_risk
    
    def _calculate_exposure_windows(self, exposures: EnvironmentalExposures,
                                   personal: PersonalFactors) -> Dict[str, Any]:
        """
        Calculate optimal windows for outdoor activities
        """
        windows = {
            'exercise': [],
            'general_outdoor': [],
            'sun_exposure': []
        }
        
        # Exercise windows (early morning or evening when AQI lower)
        if exposures.aqi < 100:
            windows['exercise'] = ['6:00-9:00 AM', '6:00-8:00 PM']
        elif exposures.aqi < 150:
            windows['exercise'] = ['6:00-7:00 AM']
        else:
            windows['exercise'] = ['Indoor exercise recommended']
            
        # Sun exposure for Vitamin D (10 AM - 3 PM optimal)
        if 3 <= exposures.uv_index <= 7:
            windows['sun_exposure'] = ['10:00 AM-3:00 PM (10-15 minutes)']
        elif exposures.uv_index < 3:
            windows['sun_exposure'] = ['Extended exposure needed for Vitamin D']
        else:
            windows['sun_exposure'] = ['Brief morning/evening exposure only']
            
        # General outdoor activities
        if exposures.aqi < 100 and exposures.pollen_count < 150:
            windows['general_outdoor'] = ['Most of day suitable']
        else:
            windows['general_outdoor'] = ['Limit to essential activities']
            
        return windows
    
    def _determine_overall_risk(self, risk_components: Dict) -> HealthRiskLevel:
        """Determine overall environmental health risk"""
        
        risk_levels = []
        
        for component in risk_components.values():
            if isinstance(component.get('risk_level'), HealthRiskLevel):
                risk_levels.append(component['risk_level'])
                
        if not risk_levels:
            return HealthRiskLevel.LOW
            
        # Take the highest risk level
        risk_values = [list(HealthRiskLevel).index(r) for r in risk_levels]
        max_risk_index = max(risk_values)
        
        return list(HealthRiskLevel)[max_risk_index]
    
    def _generate_protective_actions(self, overall_risk: HealthRiskLevel,
                                    risk_components: Dict,
                                    personal: PersonalFactors) -> List[str]:
        """Generate prioritized protective actions"""
        
        actions = []
        
        # Universal actions for moderate+ risk
        if overall_risk.value >= HealthRiskLevel.MODERATE.value:
            actions.append("Monitor symptoms closely")
            actions.append("Stay hydrated")
            
        # Air quality specific
        air_risk = risk_components.get('air_quality', {}).get('risk_level')
        if air_risk and air_risk.value >= HealthRiskLevel.HIGH.value:
            actions.append("Wear N95 mask for essential outdoor activities")
            actions.append("Run air purifier on high setting")
            
        # UV specific
        uv_risk = risk_components.get('uv_radiation', {}).get('risk_level')
        if uv_risk and uv_risk.value >= HealthRiskLevel.HIGH.value:
            actions.append("Apply SPF 30+ sunscreen 30 min before going outside")
            actions.append("Wear protective clothing and wide-brimmed hat")
            
        # Allergen specific
        if risk_components.get('allergens', {}).get('inflammation_trigger'):
            actions.append("Take antihistamine prophylactically")
            actions.append("Rinse nasal passages with saline")
            
        return actions[:5]  # Return top 5 actions
    
    def _calculate_cumulative_burden(self, exposures: EnvironmentalExposures,
                                    personal: PersonalFactors) -> float:
        """
        Calculate cumulative environmental burden score
        Based on cumulative risk assessment framework (EPA, 2003)
        """
        burden = 0.0
        
        # Air pollution burden (weighted by exposure time)
        air_burden = (exposures.pm25 / 35) * personal.outdoor_activity_hours / 8
        burden += min(1.0, air_burden) * 30  # 30% weight
        
        # UV burden (considering protection)
        uv_burden = exposures.uv_index / 11
        if not personal.uses_sunscreen_regularly:
            uv_burden *= 1.5
        burden += min(1.0, uv_burden) * 20  # 20% weight
        
        # Allergen burden
        if personal.pollen_allergy:
            pollen_burden = exposures.pollen_count / 1500
            burden += min(1.0, pollen_burden) * 25  # 25% weight
            
        # Indoor burden
        indoor_burden = 0
        if exposures.indoor_co2 > 1000:
            indoor_burden += 0.3
        if exposures.indoor_humidity < 30 or exposures.indoor_humidity > 60:
            indoor_burden += 0.3
        if abs(exposures.indoor_temp - 70) > 5:
            indoor_burden += 0.2
        burden += min(1.0, indoor_burden) * 25  # 25% weight
        
        # Apply personal vulnerability multiplier
        burden *= personal.calculate_vulnerability_score()
        
        return min(100, burden)  # Return as percentage

class EnvironmentalHealthAPI:
    """
    API interface for environmental health data integration
    Supports various environmental data sources
    """
    
    def __init__(self):
        self.model = IntegratedEnvironmentalHealthModel()
        self.data_sources = {
            'air_quality': 'EPA AirNow API',
            'uv_index': 'EPA UV Index API',
            'pollen': 'National Allergy Bureau',
            'weather': 'NOAA/NWS API'
        }
        
    def get_current_conditions(self, lat: float, lon: float) -> EnvironmentalExposures:
        """
        Fetch current environmental conditions for location
        NOTE: In production, this would make actual API calls
        """
        # Simulated data for demonstration
        import random
        
        current_conditions = EnvironmentalExposures(
            pm25=random.uniform(5, 50),
            pm10=random.uniform(10, 100),
            ozone=random.uniform(0.02, 0.08),
            aqi=random.randint(20, 150),
            uv_index=random.uniform(1, 10),
            temperature=random.uniform(60, 90),
            humidity=random.uniform(30, 70),
            pollen_count=random.randint(10, 1500),
            indoor_temp=random.uniform(65, 75),
            indoor_humidity=random.uniform(35, 55),
            indoor_co2=random.uniform(400, 1500)
        )
        
        # Calculate AQI from PM2.5
        current_conditions.aqi = current_conditions.calculate_aqi_from_pm25()
        
        return current_conditions
    
    def get_forecast(self, lat: float, lon: float, days: int = 7) -> List[Dict]:
        """
        Get environmental health forecast
        """
        forecast = []
        
        for day in range(days):
            conditions = self.get_current_conditions(lat, lon)
            
            # Simulate daily variation
            conditions.pm25 *= (0.8 + random.random() * 0.4)
            conditions.uv_index = max(0, min(11, conditions.uv_index + random.uniform(-2, 2)))
            conditions.pollen_count = int(conditions.pollen_count * (0.7 + random.random() * 0.6))
            
            forecast.append({
                'date': (datetime.now() + timedelta(days=day)).date().isoformat(),
                'conditions': conditions,
                'health_alerts': []
            })
            
        return forecast

def demonstrate_environmental_assessment():
    """
    Demonstrate the environmental health assessment system
    """
    print("ENVIRONMENTAL HEALTH RISK ASSESSMENT SYSTEM")
    print("=" * 60)
    
    # Initialize API
    api = EnvironmentalHealthAPI()
    
    # Test Case 1: High pollution day with sensitive individual
    print("\nSCENARIO 1: High Pollution Day - Asthmatic Individual")
    print("-" * 50)
    
    conditions_1 = EnvironmentalExposures(
        pm25=45.0,  # Unhealthy
        pm10=80.0,
        ozone=0.08,
        aqi=0,  # Will be calculated
        uv_index=7.0,
        temperature=85.0,
        humidity=65.0,
        pollen_count=800,
        indoor_temp=72.0,
        indoor_humidity=45.0,
        indoor_co2=900.0
    )
    conditions_1.aqi = conditions_1.calculate_aqi_from_pm25()
    
    person_1 = PersonalFactors(
        age=35,
        gender="female",
        asthma=True,
        pollen_allergy=True,
        outdoor_activity_hours=3.0,
        uses_air_purifier=False
    )
    
    assessment_1 = api.model.comprehensive_assessment(conditions_1, person_1)
    
    print(f"AQI: {conditions_1.aqi} ({assessment_1['risk_components']['air_quality']['category']})")
    print(f"Overall Risk Level: {assessment_1['overall_risk'].name}")
    print(f"Inflammation Increase: {assessment_1['risk_components']['air_quality']['inflammation_increase']:.1f}%")
    print(f"Cumulative Burden Score: {assessment_1['cumulative_burden']:.1f}/100")
    print("\nTop 3 Protective Actions:")
    for i, action in enumerate(assessment_1['protective_actions'][:3], 1):
        print(f"  {i}. {action}")
    
    # Test Case 2: Optimal conditions
    print("\nSCENARIO 2: Good Environmental Conditions")
    print("-" * 50)
    
    conditions_2 = EnvironmentalExposures(
        pm25=8.0,  # Good
        pm10=20.0,
        ozone=0.04,
        aqi=0,
        uv_index=4.0,
        temperature=72.0,
        humidity=45.0,
        pollen_count=50,
        indoor_temp=68.0,
        indoor_humidity=45.0,
        indoor_co2=600.0
    )
    conditions_2.aqi = conditions_2.calculate_aqi_from_pm25()
    
    person_2 = PersonalFactors(
        age=28,
        gender="female",
        outdoor_activity_hours=2.0,
        skin_type=3
    )
    
    assessment_2 = api.model.comprehensive_assessment(conditions_2, person_2)
    
    print(f"AQI: {conditions_2.aqi} ({assessment_2['risk_components']['air_quality']['category']})")
    print(f"Overall Risk Level: {assessment_2['overall_risk'].name}")
    print(f"UV Index: {conditions_2.uv_index}")
    print(f"Vitamin D Synthesis Sufficient: {assessment_2['risk_components']['uv_radiation']['vitamin_d_status']['sufficient']}")
    print("\nOptimal Exposure Windows:")
    for activity, windows in assessment_2['exposure_windows'].items():
        print(f"  {activity}: {', '.join(windows) if windows else 'Not recommended'}")
    
    # Test Case 3: Indoor environment issues
    print("\nSCENARIO 3: Poor Indoor Air Quality")
    print("-" * 50)
    
    conditions_3 = EnvironmentalExposures(
        pm25=12.0,
        aqi=50,
        indoor_temp=78.0,  # Too warm
        indoor_humidity=65.0,  # Too humid
        indoor_co2=1800.0,  # High CO2
        indoor_voc=0.8
    )
    
    person_3 = PersonalFactors(
        age=45,
        gender="female"
    )
    
    assessment_3 = api.model.comprehensive_assessment(conditions_3, person_3)
    
    print(f"Indoor CO2: {conditions_3.indoor_co2} ppm")
    print(f"Indoor Humidity: {conditions_3.indoor_humidity}%")
    print(f"Indoor Environment Quality: {assessment_3['risk_components']['indoor_environment']['overall_quality']}")
    print("\nHealth Impacts:")
    all_impacts = (assessment_3['health_impacts']['immediate'] + 
                  assessment_3['health_impacts']['short_term'])
    for impact in all_impacts[:3]:
        print(f"  • {impact}")
    
    # Generate 7-day forecast
    print("\n7-DAY ENVIRONMENTAL HEALTH FORECAST")
    print("-" * 50)
    
    forecast = api.get_forecast(40.7128, -74.0060, days=7)
    
    print("Date       | AQI  | UV  | Pollen | Risk Level")
    print("-" * 50)
    for day_forecast in forecast:
        conditions = day_forecast['conditions']
        assessment = api.model.comprehensive_assessment(conditions, person_2)
        print(f"{day_forecast['date']} | {conditions.aqi:3d}  | {conditions.uv_index:.1f} | "
              f"{conditions.pollen_count:6d} | {assessment['overall_risk'].name}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("\nNOTE: This model provides health risk assessment based on")
    print("environmental factors. Always consult healthcare providers")
    print("for medical advice and follow official health warnings.")

if __name__ == "__main__":
    demonstrate_environmental_assessment()

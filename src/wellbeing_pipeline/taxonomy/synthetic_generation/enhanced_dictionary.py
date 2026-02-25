"""
Enhanced Dictionary Generator for API Context Understanding

Purpose:
- Transform Dictionary_v2.json into an API-friendly format
- Add business context examples for each category
- Provide phrase templates for realistic combinations
- Map to theme/subtheme structure for better alignment
- Include sentiment/polarity guidance

Output:
- Enhanced_Dictionary_v3.json (API-optimized)
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from ...config_runtime import _get_project_root

# Paths
PROJECT_ROOT = _get_project_root(start=Path(__file__).resolve())
CONFIG_DIR = PROJECT_ROOT / "config"
ASSETS_DIR = PROJECT_ROOT / "assets" / "taxonomy"
INPUT_FILE = CONFIG_DIR / "Dictionary_v2.json"
OUTPUT_FILE = ASSETS_DIR / "Enhanced_Dictionary_v3.json"

def load_dictionary_v2() -> Dict:
    """Load the original Dictionary_v2.json"""
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_enhanced_dictionary() -> Dict:
    """
    Create an enhanced dictionary with:
    1. Business context examples
    2. Phrase templates
    3. Theme/subtheme mappings
    4. Sentiment guidance
    5. Column-specific usage
    """
    
    original = load_dictionary_v2()
    
    enhanced = {
        "_metadata": {
            "version": "3.0",
            "description": "Enhanced dictionary optimized for API context understanding and phrase generation",
            "source": "Dictionary_v2.json",
            "enhancements": [
                "Added business context examples per category",
                "Added phrase templates for realistic combinations",
                "Added theme/subtheme mappings",
                "Added sentiment/polarity guidance",
                "Added column-specific usage examples",
                "Structured for LLM consumption"
            ]
        },
        
        # ========================================
        # BUSINESS CONTEXT CATEGORIES
        # ========================================
        "business_context": {
            "projects": {
                "description": "Project-related terms, programmes, and initiatives",
                "acronyms": ["amp", "amp7", "amp8", "pmo", "prj"],
                "terms": ["programme", "project", "alliance"],
                "example_phrases": {
                    "Wellbeing_Details": [
                        "amp8 uncertainty causing stress",
                        "amp8 delays affecting workload",
                        "pmo communication issues",
                        "project deadlines too tight",
                        "alliance restructure anxiety",
                        "programme changes unclear",
                        "amp7 transition problems",
                        "project pressure increasing"
                    ],
                    "Areas_Improve": [
                        "need clearer amp8 communication",
                        "better pmo updates required",
                        "more project planning time",
                        "improve alliance coordination",
                        "clearer programme direction needed",
                        "better project resource allocation"
                    ],
                    "Support_Provided": [
                        "project team supportive",
                        "pmo providing regular updates",
                        "alliance colleagues helpful",
                        "programme manager accessible"
                    ]
                },
                "phrase_templates": [
                    "{project_term} {issue_word}",
                    "{project_term} causing {stress_word}",
                    "lack of {project_term} {clarity_word}",
                    "need better {project_term} {support_word}"
                ],
                "related_themes": [
                    "Organisational Change & Leadership Decisions",
                    "Job Security & Programme Uncertainty",
                    "Workload & Pressure"
                ]
            },
            
            "management": {
                "description": "Management, leadership, and supervision terms",
                "acronyms": ["ipl", "pm"],
                "terms": ["line manager", "supervisor", "leadership", "team lead", "director", "senior", "site manager"],
                "example_phrases": {
                    "Wellbeing_Details": [
                        "line manager not supportive",
                        "ipl communication poor",
                        "lack of leadership direction",
                        "senior management decisions unclear",
                        "supervisor doesn't listen",
                        "team lead unavailable",
                        "site manager expectations unrealistic",
                        "poor management communication"
                    ],
                    "Areas_Improve": [
                        "need better line manager support",
                        "more ipl communication required",
                        "clearer leadership direction",
                        "better senior management updates",
                        "more supervisor feedback",
                        "improved team lead accessibility"
                    ],
                    "Support_Provided": [
                        "line manager very supportive",
                        "ipl always available",
                        "leadership team accessible",
                        "senior management transparent",
                        "supervisor provides guidance",
                        "team lead helpful"
                    ]
                },
                "phrase_templates": [
                    "{manager_term} {support_word}",
                    "{manager_term} {communication_word}",
                    "lack of {manager_term} {quality_word}",
                    "{manager_term} doesn't {action_word}"
                ],
                "related_themes": [
                    "Management & Leadership",
                    "Communication",
                    "Team & Colleagues"
                ]
            },
            
            "workload": {
                "description": "Workload, capacity, and pressure terms",
                "terms": ["backlog", "workload", "pressure", "capacity", "resourcing", "volume", "targets", "working hours", "working away"],
                "example_phrases": {
                    "Wellbeing_Details": [
                        "workload too high",
                        "backlog overwhelming",
                        "constant pressure",
                        "capacity issues",
                        "resourcing problems",
                        "volume unmanageable",
                        "targets unrealistic",
                        "working hours excessive",
                        "working away too much",
                        "work load causing stress"
                    ],
                    "Areas_Improve": [
                        "need better workload distribution",
                        "reduce backlog",
                        "more capacity required",
                        "better resourcing needed",
                        "realistic targets",
                        "reduce working hours",
                        "less working away"
                    ],
                    "Support_Provided": [
                        "workload manageable",
                        "team helps with backlog",
                        "capacity adequate",
                        "resourcing improved"
                    ]
                },
                "phrase_templates": [
                    "{workload_term} too {intensity_word}",
                    "{workload_term} causing {stress_word}",
                    "need better {workload_term} {management_word}",
                    "{workload_term} {issue_word}"
                ],
                "related_themes": [
                    "Workload & Pressure",
                    "Work–Life Balance",
                    "Stress & Burnout"
                ]
            },
            
            "operational": {
                "description": "Operational and field work terms",
                "acronyms": ["dma"],
                "terms": ["site job", "planned", "reactive", "on-site", "onsite", "field"],
                "example_phrases": {
                    "Wellbeing_Details": [
                        "site job conditions poor",
                        "reactive work stressful",
                        "on-site facilities inadequate",
                        "field work demanding",
                        "dma issues constant",
                        "planned work disrupted"
                    ],
                    "Areas_Improve": [
                        "better site job planning",
                        "reduce reactive work",
                        "improve on-site facilities",
                        "better field work support",
                        "clearer dma processes"
                    ],
                    "Support_Provided": [
                        "site job team supportive",
                        "field work colleagues helpful",
                        "on-site facilities adequate"
                    ]
                },
                "phrase_templates": [
                    "{operational_term} {issue_word}",
                    "{operational_term} causing {stress_word}",
                    "need better {operational_term} {support_word}"
                ],
                "related_themes": [
                    "Workload & Pressure",
                    "Physical Health",
                    "Support (Workplace)"
                ]
            },
            
            "office": {
                "description": "Office location and workspace terms",
                "acronyms": ["twh", "hq"],
                "terms": ["head office", "desk space", "hotdesk"],
                "example_phrases": {
                    "Wellbeing_Details": [
                        "twh desk space limited",
                        "hotdesking stressful",
                        "head office too far",
                        "desk space inadequate",
                        "hq facilities poor"
                    ],
                    "Areas_Improve": [
                        "more desk space at twh",
                        "better hotdesking system",
                        "improve head office facilities",
                        "dedicated desk space needed"
                    ],
                    "Support_Provided": [
                        "twh facilities good",
                        "hotdesking works well",
                        "head office accessible",
                        "desk space adequate"
                    ]
                },
                "phrase_templates": [
                    "{office_term} {issue_word}",
                    "lack of {office_term} {resource_word}",
                    "{office_term} {quality_word}"
                ],
                "related_themes": [
                    "Support (Workplace)",
                    "Physical Health",
                    "Work–Life Balance"
                ]
            },
            
            "partner_companies": {
                "description": "Partner and contractor company names",
                "companies": ["balfour beatty", "barhale", "binnies", "mott macdonald bentley", "mmb", "mwh", "skanska", "sweco"],
                "example_phrases": {
                    "Wellbeing_Details": [
                        "mmb communication issues",
                        "balfour beatty coordination problems",
                        "barhale expectations unclear",
                        "mwh transition difficult",
                        "skanska integration challenging"
                    ],
                    "Areas_Improve": [
                        "better mmb coordination",
                        "clearer balfour beatty communication",
                        "improve barhale processes"
                    ],
                    "Support_Provided": [
                        "mmb team supportive",
                        "balfour beatty colleagues helpful",
                        "barhale management accessible"
                    ]
                },
                "phrase_templates": [
                    "{company} {issue_word}",
                    "{company} {communication_word}",
                    "{company} team {quality_word}"
                ],
                "related_themes": [
                    "Team & Colleagues",
                    "Communication",
                    "Management & Leadership"
                ]
            },
            
            "systems": {
                "description": "IT systems and tools",
                "acronyms": ["sap", "ecc", "ctc", "tdw", "bs", "cm", "hrs", "aws"],
                "example_phrases": {
                    "Wellbeing_Details": [
                        "sap implementation stressful",
                        "sap training inadequate",
                        "sap system issues",
                        "ecc problems constant",
                        "hrs system difficult",
                        "lack of sap support"
                    ],
                    "Areas_Improve": [
                        "need sap training",
                        "better sap support required",
                        "improve ecc system",
                        "more hrs training needed",
                        "clearer sap guidance"
                    ],
                    "Support_Provided": [
                        "sap team helpful",
                        "sap training provided",
                        "ecc support available"
                    ]
                },
                "phrase_templates": [
                    "{system} {issue_word}",
                    "{system} causing {stress_word}",
                    "lack of {system} {training_word}",
                    "need better {system} {support_word}"
                ],
                "related_themes": [
                    "Career Development & Training",
                    "Support (Workplace)",
                    "Workload & Pressure"
                ]
            },
            
            "benefits": {
                "description": "Benefits and compensation terms",
                "terms": ["cash cars", "benefits", "pay", "salary", "bonus", "expenses"],
                "example_phrases": {
                    "Wellbeing_Details": [
                        "cash cars policy unclear",
                        "benefits inadequate",
                        "pay concerns",
                        "salary not competitive",
                        "bonus structure unfair",
                        "expenses process difficult"
                    ],
                    "Areas_Improve": [
                        "clearer cash cars policy",
                        "better benefits package",
                        "improve pay structure",
                        "review salary levels",
                        "fairer bonus system",
                        "simplify expenses process"
                    ],
                    "Support_Provided": [
                        "cash cars scheme helpful",
                        "benefits package good",
                        "pay competitive",
                        "expenses processed quickly"
                    ]
                },
                "phrase_templates": [
                    "{benefit_term} {issue_word}",
                    "{benefit_term} not {quality_word}",
                    "need better {benefit_term}"
                ],
                "related_themes": [
                    "Financial Concerns",
                    "Pay & Salary Satisfaction",
                    "Benefits & Compensation"
                ]
            },
            
            "support": {
                "description": "Support systems and resources",
                "terms": ["child care", "family friends", "support", "resources"],
                "example_phrases": {
                    "Wellbeing_Details": [
                        "lack of child care support",
                        "family friends support limited",
                        "support resources inadequate"
                    ],
                    "Areas_Improve": [
                        "need child care support",
                        "better support resources",
                        "more family-friendly policies"
                    ],
                    "Support_Provided": [
                        "child care support available",
                        "family friends supportive",
                        "support resources helpful"
                    ]
                },
                "phrase_templates": [
                    "{support_term} {availability_word}",
                    "lack of {support_term}",
                    "{support_term} helpful"
                ],
                "related_themes": [
                    "Support (Personal)",
                    "Support (Workplace)",
                    "Work–Life Balance"
                ]
            },
            
            "health": {
                "description": "Health and wellbeing terms",
                "terms": ["back pain", "chronic pain", "mental health", "physical health", "wellbeing", "stress"],
                "example_phrases": {
                    "Wellbeing_Details": [
                        "back pain from work",
                        "chronic pain affecting work",
                        "mental health struggling",
                        "physical health declining",
                        "wellbeing poor",
                        "stress levels high"
                    ],
                    "Areas_Improve": [
                        "need back pain support",
                        "better mental health resources",
                        "improve physical health support",
                        "more wellbeing initiatives",
                        "reduce stress levels"
                    ],
                    "Support_Provided": [
                        "mental health support available",
                        "wellbeing initiatives helpful",
                        "physical health resources good"
                    ]
                },
                "phrase_templates": [
                    "{health_term} {issue_word}",
                    "{health_term} affecting {work_word}",
                    "need {health_term} {support_word}"
                ],
                "related_themes": [
                    "Mental Health & Wellbeing",
                    "Physical Health",
                    "Stress & Burnout"
                ]
            }
        },
        
        # ========================================
        # PHRASE BUILDING BLOCKS
        # ========================================
        "phrase_building_blocks": {
            "issue_words": ["issues", "problems", "concerns", "difficulties", "challenges"],
            "stress_words": ["stress", "anxiety", "worry", "pressure", "strain"],
            "clarity_words": ["clarity", "information", "updates", "communication", "direction"],
            "support_words": ["support", "help", "assistance", "guidance", "resources"],
            "quality_words": ["quality", "standard", "level", "effectiveness"],
            "communication_words": ["communication", "updates", "information", "feedback"],
            "action_words": ["listen", "communicate", "support", "understand", "help"],
            "intensity_words": ["high", "excessive", "overwhelming", "unmanageable"],
            "management_words": ["management", "distribution", "allocation", "planning"],
            "resource_words": ["resources", "facilities", "space", "equipment"],
            "training_words": ["training", "guidance", "support", "instruction"],
            "availability_words": ["available", "accessible", "provided", "offered"],
            "work_words": ["work", "job", "role", "duties", "tasks"]
        },
        
        # ========================================
        # COLUMN-SPECIFIC GUIDANCE
        # ========================================
        "column_guidance": {
            "Wellbeing_Details": {
                "description": "Problems, issues, stressors affecting wellbeing",
                "tone": "negative, problem-focused",
                "typical_patterns": [
                    "{term} causing {stress_word}",
                    "lack of {term}",
                    "{term} {issue_word}",
                    "{term} too {intensity_word}",
                    "{term} not {quality_word}"
                ],
                "example_starters": [
                    "lack of",
                    "no",
                    "poor",
                    "inadequate",
                    "too much",
                    "not enough",
                    "causing stress",
                    "affecting"
                ]
            },
            "Areas_Improve": {
                "description": "Improvement requests, suggestions, asks",
                "tone": "constructive, solution-focused",
                "typical_patterns": [
                    "need {term}",
                    "better {term} required",
                    "more {term}",
                    "improve {term}",
                    "clearer {term}"
                ],
                "example_starters": [
                    "need",
                    "require",
                    "want",
                    "would like",
                    "better",
                    "more",
                    "improve",
                    "clearer"
                ]
            },
            "Support_Provided": {
                "description": "Positive supports, buffers, coping mechanisms",
                "tone": "positive, appreciative",
                "typical_patterns": [
                    "{term} helpful",
                    "{term} supportive",
                    "{term} good",
                    "{term} available"
                ],
                "example_starters": [
                    "helpful",
                    "supportive",
                    "good",
                    "available",
                    "accessible",
                    "provided",
                    "works well"
                ]
            }
        },
        
        # ========================================
        # PRESERVE ORIGINAL DICTIONARY RULES
        # ========================================
        "original_rules": {
            "domain_whitelist": original.get("DOMAIN_WHITELIST", []),
            "business_map": original.get("BUSINESS_MAP", {}),
            "unigram_whitelist": original.get("UNIGRAM_WHITELIST", []),
            "always_drop_unigrams": original.get("ALWAYS_DROP_UNIGRAMS", []),
            "must_be_phrase": original.get("MUST_BE_PHRASE", []),
            "force_phrases": original.get("FORCE_PHRASES", []),
            "merge_to_canonical": original.get("MERGE_TO_CANONICAL", {})
        }
    }
    
    return enhanced

def main():
    print("="*80)
    print("ENHANCED DICTIONARY GENERATOR")
    print("="*80)
    
    print(f"\n📄 Loading: {INPUT_FILE}")
    enhanced = create_enhanced_dictionary()
    
    print(f"\n💾 Saving: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(enhanced, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n✅ Enhanced Dictionary Created!")
    print(f"\n📊 Summary:")
    print(f"   • Business categories: {len(enhanced['business_context'])}")
    
    total_examples = sum(
        len(cat['example_phrases'].get('Wellbeing_Details', [])) +
        len(cat['example_phrases'].get('Areas_Improve', [])) +
        len(cat['example_phrases'].get('Support_Provided', []))
        for cat in enhanced['business_context'].values()
        if 'example_phrases' in cat
    )
    print(f"   • Total example phrases: {total_examples}")
    print(f"   • Phrase building blocks: {len(enhanced['phrase_building_blocks'])}")
    print(f"   • Column guidance: {len(enhanced['column_guidance'])}")
    
    print(f"\n💡 This enhanced dictionary provides:")
    print(f"   ✓ Business context examples for each category")
    print(f"   ✓ Phrase templates for realistic combinations")
    print(f"   ✓ Column-specific usage guidance")
    print(f"   ✓ Theme/subtheme mappings")
    print(f"   ✓ Sentiment/polarity guidance")
    print(f"\n🎯 Ready for API consumption in phrase generation!")
    print("="*80)

if __name__ == "__main__":
    main()

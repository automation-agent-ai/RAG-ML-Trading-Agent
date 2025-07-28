#!/usr/bin/env python3
"""
News Categories and Patterns

This file contains the definitions of news categories and patterns
for the news classification system.
"""

# RNS Category to Group Mapping System
RNS_CATEGORIES = [
    "AGM Statement", "Admission of Securities", "Annual Financial Report", "Blocklisting",
    "Board Changes", "Broker Recommendation", "Capital Reorganisation", "Change of Adviser",
    "Change of Name", "Circular", "Company Secretary Change", "Contract Award",
    "Contract Termination", "Conversion of Securities", "Credit Rating", "Debt Issue",
    "Delisting", "Director Dealings", "Director/PDMR Shareholding", "Dividend Declaration",
    "Final Results", "Interim Results", "Investment Update", "Issue of Equity",
    "Launch of New Product", "Major Interest in Shares", "Merger & Acquisition",
    "Notice of AGM", "Placing", "Profit Warning", "Resignation", "Results of AGM",
    "Share Buyback", "Shareholder Meeting", "Trading Statement", "Trading Update"
]

CATEGORY_TO_GROUP = {
    # Financial Results
    "Final Results": "financial_results",
    "Interim Results": "financial_results", 
    "Profit Warning": "financial_results",
    "Trading Statement": "financial_results",
    "Trading Update": "financial_results",
    "Annual Financial Report": "financial_results",
    
    # Corporate Actions
    "Share Buyback": "corporate_actions",
    "Issue of Equity": "corporate_actions",
    "Dividend Declaration": "corporate_actions",
    "Capital Reorganisation": "corporate_actions",
    "Placing": "corporate_actions",
    "Debt Issue": "corporate_actions",
    "Admission of Securities": "corporate_actions",
    "Conversion of Securities": "corporate_actions",
    "Blocklisting": "corporate_actions",
    "Delisting": "corporate_actions",
    
    # Governance
    "Board Changes": "governance",
    "Director Dealings": "governance",
    "Director/PDMR Shareholding": "governance",
    "AGM Statement": "governance",
    "Results of AGM": "governance",
    "Notice of AGM": "governance",
    "Shareholder Meeting": "governance",
    "Company Secretary Change": "governance",
    "Resignation": "governance",
    "Major Interest in Shares": "governance",
    "Change of Name": "governance",
    "Change of Adviser": "governance",
    "Circular": "governance",
    
    # Corporate Events
    "Contract Award": "corporate_events",
    "Merger & Acquisition": "corporate_events",
    "Launch of New Product": "corporate_events",
    "Contract Termination": "corporate_events",
    
    # Other Signals
    "Broker Recommendation": "other_signals",
    "Credit Rating": "other_signals",
    "Investment Update": "other_signals"
}

# Pattern matching dictionaries for fast classification
FINANCIAL_RESULTS_PATTERNS = {
    "final results": "Final Results",
    "interim results": "Interim Results",
    "profit warning": "Profit Warning",
    "trading statement": "Trading Statement",
    "trading update": "Trading Update",
    "annual financial report": "Annual Financial Report",
    "annual report": "Annual Financial Report",
    "financial results": "Final Results",
    "half year results": "Interim Results",
    "half-year results": "Interim Results",
    "full year results": "Final Results",
    "full-year results": "Final Results",
    "preliminary results": "Final Results",
    "audited results": "Final Results",
    "unaudited results": "Interim Results",
}

CORPORATE_ACTIONS_PATTERNS = {
    "share buyback": "Share Buyback",
    "issue of equity": "Issue of Equity",
    "dividend declaration": "Dividend Declaration",
    "capital reorganisation": "Capital Reorganisation",
    "placing": "Placing",
    "debt issue": "Debt Issue",
    "admission of securities": "Admission of Securities",
    "conversion of securities": "Conversion of Securities",
    "blocklisting": "Blocklisting",
    "delisting": "Delisting",
    "share repurchase": "Share Buyback",
    "dividend announcement": "Dividend Declaration",
    "capital restructuring": "Capital Reorganisation",
    "equity placing": "Placing",
    "bond issue": "Debt Issue",
    "share issue": "Issue of Equity",
}

GOVERNANCE_PATTERNS = {
    "board changes": "Board Changes",
    "director dealings": "Director Dealings",
    "director shareholding": "Director/PDMR Shareholding",
    "pdmr shareholding": "Director/PDMR Shareholding",
    "agm statement": "AGM Statement",
    "results of agm": "Results of AGM",
    "notice of agm": "Notice of AGM",
    "shareholder meeting": "Shareholder Meeting",
    "company secretary change": "Company Secretary Change",
    "resignation": "Resignation",
    "major interest in shares": "Major Interest in Shares",
    "change of name": "Change of Name",
    "change of adviser": "Change of Adviser",
    "circular": "Circular",
    "board appointment": "Board Changes",
    "director appointment": "Board Changes",
    "ceo appointment": "Board Changes",
    "cfo appointment": "Board Changes",
    "chairman appointment": "Board Changes",
    "board resignation": "Resignation",
    "director resignation": "Resignation",
    "ceo resignation": "Resignation",
    "cfo resignation": "Resignation",
    "chairman resignation": "Resignation",
}

CORPORATE_EVENTS_PATTERNS = {
    "contract award": "Contract Award",
    "merger": "Merger & Acquisition",
    "acquisition": "Merger & Acquisition",
    "launch of new product": "Launch of New Product",
    "contract termination": "Contract Termination",
    "new contract": "Contract Award",
    "m&a": "Merger & Acquisition",
    "product launch": "Launch of New Product",
    "contract win": "Contract Award",
    "major contract": "Contract Award",
}

OTHER_SIGNALS_PATTERNS = {
    "broker recommendation": "Broker Recommendation",
    "credit rating": "Credit Rating",
    "investment update": "Investment Update",
    "analyst recommendation": "Broker Recommendation",
    "rating change": "Credit Rating",
    "investor update": "Investment Update",
    "market update": "Investment Update",
    "operational update": "Investment Update",
} 
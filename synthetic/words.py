"""
Word lists shared by the generators. Short names keep repeated characters few.
"""

FIRST = [
    "alex", "amy", "ben", "beth", "carl", "chris", "dan", "eric", "fay", "greg",
    "hugo", "ian", "ivy", "jack", "jake", "jon", "jude", "kate", "kim", "liam",
    "lucy", "luke", "mark", "max", "mia", "mike", "nick", "nora", "owen", "paul",
    "rob", "ruby", "sam", "tom", "troy", "zoe",
]  # fmt: skip

LAST = [
    "black", "brown", "clark", "cruz", "diaz", "doyle", "ford", "fox", "grant",
    "hart", "king", "knight", "lewis", "lynch", "marsh", "moss", "park", "pike",
    "price", "quinn", "shaw", "stark", "stone", "vance", "ward", "webb", "wolf",
    "young",
]  # fmt: skip

COMPANIES = [
    "acme", "apex", "brisk", "clay", "dusk", "fjord", "glyph", "helix", "jolt",
    "kite", "lumen", "onyx", "pixel", "quartz", "rift", "sync", "vortex",
]  # fmt: skip

PERSONAL_DOMAINS = [
    "gmail.com", "yahoo.com", "icloud.com", "aol.com", "proton.me", "gmx.de",
    "web.de", "mail.ru", "me.com", "pm.me", "hey.com", "sky.com", "live.com",
    "msn.com", "zoho.com",
]  # fmt: skip

COMPANY_DOMAINS = [
    f"{company}.{tld}" for company in COMPANIES for tld in ("com", "io", "co", "dev")
]

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    "January", "February", "March", "April", "June", "July", "August", "September",
    "October", "November", "December",
]  # fmt: skip

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

ACRONYMS = [
    "AI", "API", "B2B", "CEO", "CFO", "CRM", "CTO", "FAQ", "GDPR", "HQ", "ISO",
    "KPI", "MVP", "OKR", "PS", "Q&A", "QR", "SaaS", "SEO", "SME", "UK", "USA",
    "UX", "VAT", "VP",
]  # fmt: skip

METRICS = ["ROI", "NPS", "ARR", "MRR", "EBITDA", "revenue", "growth", "churn"]

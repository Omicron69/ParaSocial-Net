"""
Behavioural lexicons for Phase A feature extraction.

Each domain holds groups of terms; features count how densely a user's text
hits each group. Matching is substring-based on lowercased text, so base forms
(bias, oshi), OOV variants (biases, ults) and native scripts (Hangul, Kana,
Kanji) all match without tokenisation concerns.

Scope: K-pop and J-pop English-language communities. Terms are included only if
they plausibly appear in the corpus and signal the behaviour. Ambiguous bare
English words that collide with ordinary text (my, once, go, anti, live) are
deliberately excluded to keep features discriminative.
"""

# ----------------------------------------------------------------------------
# PARASOCIAL — attachment intensity, identity fusion, fan-collective identity
# ----------------------------------------------------------------------------
PARASOCIAL = {
    "attachment": [
        "bias", "biases", "ult", "ults", "ultimate bias", "bias wrecker",
        "wrecker", "delulu", "delusional", "lovesick", "obsessed", "obsession",
        "stan", "stanning", "stanned", "hard stan", "soft stan", "multi stan",
        "multistan", "comfort person", "comfort character", "hyperfixation",
        "parasocial", "solo stan", "all-member stan",
        "최애", "차애", "악개", "올팬",
        "oshi", "oshimen", "my oshi", "kami-oshi", "gachikoi", "gachi-koi",
        "wotaku", "hako-oshi", "oshihen", "kamiban", "ichiban",
        "tantou", "doutan", "推し", "推しメン", "神推し",
        "ガチ恋", "ヲタク", "箱推し", "推し変", "誰でも大好き", "担当",
        "同担", "celebrity crush", "blorbo", "idolize", "stan account",
    ],
    "fusion": [
        "my everything", "my whole world", "reason to live",
        "saved me", "saved my life", "would do anything",
        "owns my heart", "owns my soul", "can't live without", "cant live without",
        "my reason for breathing", "would die for", "protect at all costs",
        "must protect", "my husband", "my wife", "wifey",
        "my man", "my girl", "my boyfriend", "my girlfriend",
        "oppa", "unnie", "noona", "goddess", "my queen", "my king",
        "my angel", "오빠", "언니", "누나",
        "사랑해", "내꺼", "보고 싶어", "yome", "danna", "嫁", "旦那", "愛してる",
        "尊い", "mother is mothering", "my roman empire",
        "comfort human", "she is the moment", "he is the moment",
        "whole personality",
    ],
    "collective": [
        "my fandom", "we as fans", "us fans", "stan twitter", "stan twt",
        "fandom space", "fandom spaces",
        "armys", "blinks", "onces", "carats",
        "engenes", "engene", "moas", "nctzens", "nctzen", "시즈니",
        "exols", "exo-ls", "tokkis", "shawols", "atinys", "atiny",
        "reveluvs", "reveluv", "midzys", "fearnots", "orbits",
        "ouruu",
        "ohisama", "おひさま", "buddies", "バディーズ", "nogiota", "乃木オタ",
        "mononofu", "モノノフ", "haroota", "ハロヲタ", "eighter", "エイター",
        "tobikko", "とびっこ", "kazetarians", "kazetarian", "風民",
    ],
    "ritual": [
        "streaming party", "stream goal", "stream party", "mass streaming",
        "mass stream", "listening party", "listening parties",
        "album rollout", "midnight drop", "fan project",
        "weverse", "위버스", "stationhead", "fansite",
        "fancam", "fancams", "choeaedol",
        "comeback", "comebacks", "컴백", "fanmeeting", "fanmeet", "팬미팅",
        "fansign", "fansigns", "팬싸", "fancall", "fan call", "영통",
        "hi-touch", "music show", "음방", "streaming", "직캠", "응원", "화이팅",
        "handshake event", "handshake ticket", "握手会", "cheki session",
        "チェキ会", "senbatsu", "選抜", "sousenkyo", "総選挙",
        "seitan-sai", "生誕祭",
        "verified fan", "presale code", "merch drop",
    ],
}

# ----------------------------------------------------------------------------
# FINANCIAL — spending, collecting, acquisition, grey-market trading
# ----------------------------------------------------------------------------
FINANCIAL = {
    "collecting": [
        "photocard", "photocards", "pcs", "polaroids", "postcards",
        "plushies", "sleeves", "freebies", "merch", "merchandise",
        "unboxing", "signed vinyl", "tour merch", "trading cards",
        "lithograph", "toploader", "binder",
        "lightstick", "lightsticks", "skzoo", "pob", "lucky draw",
        "season's greetings", "deco kit", "포카", "응원봉",
        "탑로더", "특전",
        "bromide", "ブロマイド", "acrylic stand", "aksta", "アクスタ",
        "アクリルスタンド", "uchiwa", "うちわ", "penlight", "ペンライト",
        "cheki", "チェキ", "can badge", "缶バッジ", "photobook", "写真集",
    ],
    "spend": [
        "pre-order", "pre-orders", "pre-ordered", "preorder", "preorders",
        "preordered", "pre-sale", "presale", "purchasing", "purchased",
        "repackage", "restock", "restocked", "haul",
        "copies", "versions", "resale",
        "dynamic pricing", "vip package", "bulk buy", "sold out",
        "wts", "want to sell", "wtt", "want to trade", "wtb", "want to buy",
        "in search of", "looking for buyer", "looking for seller",
        "group order", "gom", "unsealed", "sealed", "pooling",
        "양도", "공구", "분할", "원가", "미개봉",
        "lottery", "ballot", "抽選", "当落", "積む", "共同購入",
        "譲渡", "定価", "買取",
    ],
    "strain": [
        "overpriced", "in debt", "cant afford", "can't afford",
        "rent money", "food money", "maxed out",
        "credit card", "payment plan", "klarna", "afterpay",
        "price gouging", "scammed", "scammer",
        "bankrupt", "bankruptcy", "chargeback",
        "텅장", "사기", "사기꾼", "플미",
        "tenbai", "tenbaiya", "転売", "転売屋", "高値", "金欠",
        "破産", "詐欺", "借金",
    ],
}

# ----------------------------------------------------------------------------
# VICTIM — harm, harassment, coordinated conflict, victimisation
# (TIGHT: harm/conflict signal only, not generic negativity)
# ----------------------------------------------------------------------------
VICTIM = {
    "attack": [
        "harass", "harassed", "harassing", "harassment", "bully", "bullied",
        "bullying", "cyberbullying", "kys", "kill yourself", "death threats",
        "threatened", "threatening", "untalented", "hate train", "pile-on",
        "flaming", "ratioed",
        "악플", "악플러", "패드립",
        "enjou", "炎上", "bashing", "バッシング", "アンチスレ",
    ],
    "invasion": [
        "doxx", "doxxed", "doxing", "doxxing", "dropping dox",
        "personal information", "real name", "workplace",
        "stalking", "stalker", "swatting", "swatted", "ip logger",
        "sasaeng", "sasaengs", "사생", "사생팬", "신상털기",
        "yarakashi", "やらかし", "yakkai", "厄介", "tsukematoi", "つきまとい",
        "sarashi", "晒し",
    ],
    "fanwar": [
        "fanwar", "fanwars", "stan war", "stan wars", "antis", "anti-fan",
        "cancelled", "cancel culture", "mass report", "mass reporting",
        "brigade", "brigading", "gatekeeping", "boycott", "boycotting",
        "deplatform", "akgaes", "akgae", "oppalogist", "black ocean",
        "protest truck",
        "안티", "팬덤싸움", "트럭시위", "총공", "병크", "텐미닛",
        "kuro-umi", "黒海", "同担拒否", "マナー違反",
    ],
    "hostility": [
        "misogyny", "misogynistic", "xenophobia", "xenophobic", "homophobic",
        "racist", "racism", "n-word", "appropriation", "defamation",
        "allegations", "lawsuit", "lawsuits", "sexual assault",
        "ableist", "colorism", "fatphobic", "fatphobia", "legal action",
        "고소", "명예훼손", "성희롱", "외모지상주의",
        "meiyo kison", "名誉毀損", "訴訟", "sekuhara", "セクハラ", "差별",
    ],
    "targeted": [
        "came after me", "got harassed", "attacked me", "targeted me",
        "hate comments", "they doxxed", "i was doxxed", "getting hate",
        "sent me death threats", "told me to kys",
        "mass reporting me", "brigaded my",
        "저격당함", "악플 받음", "晒された", "叩かれてる",
    ],
}

# ----------------------------------------------------------------------------
# GENERAL — fandom/industry vocabulary with no behavioural signal.
# Kept for reference; NOT counted as a feature.
# ----------------------------------------------------------------------------
GENERAL = [
    "charting", "billboard", "lead single", "b-side", "b-sides",
    "title track", "mixtape", "world tour", "arena tour", "setlist", "encore",
    "lipsync", "adlibs", "choreography", "concept photo",
    "comeback", "comebacks", "daesang", "bonsang", "all-kill",
    "perfect all kill", "inkigayo", "mcountdown", "music bank", "show champion",
    "music show", "disbandment", "disbanded", "sub-unit", "subunit", "predebut",
    "pre-debut", "trainees", "netizens", "knetz", "allkpop", "soompi",
    "hanteo", "circle chart", "melon", "k-pop", "kpop", "nugu", "ending fairy",
    "컴백", "데뷔", "음원", "해체", "대상",
    "senbatsu", "akbingo", "j-pop", "jpop", "graduation", "graduated",
    "kouhaku", "oricon", "budokan", "dome tour", "johnnys", "starto", "sakamichi",
    "卒業", "選抜", "オリコン", "紅白", "武道館", "解散",
]

# ----------------------------------------------------------------------------
DOMAINS = {"parasocial": PARASOCIAL, "financial": FINANCIAL, "victim": VICTIM}


def terms(group):
    """Flatten a grouped domain to a sorted, de-duplicated term list."""
    return sorted({t for g in group.values() for t in g})


if __name__ == "__main__":
    seen = {}
    for name, d in DOMAINS.items():
        flat = terms(d)
        print(f"{name:11} {len(d)} groups, {len(flat)} terms")
        for t in flat:
            seen.setdefault(t, []).append(name)
    dupes = {t: ds for t, ds in seen.items() if len(ds) > 1}
    print("cross-domain duplicates:", dupes if dupes else "none")
"""What the rider sees and hears back for each product card (illustrative UI copy only).

src/product.py decides the card kind/primary action; it has no rider-facing strings, so these are the demo
defaults. Nothing here is executed: no call, SMS or ticket is actually placed.
"""

from __future__ import annotations

# key -> {lang: (instruction, spoken)}, gloss, buttons, status, ops
COPY = {
    "wait": {
        "text": {
            "hinglish": ("Ruko, customer ko bata diya hai", "Ruko, customer ko bata diya hai. Paanch minute ka timer chalu hai."),
            "hindi": ("रुको, कस्टमर को बता दिया है", "रुको, कस्टमर को बता दिया है। पाँच मिनट का टाइमर चालू है।"),
            "english": ("Hold on, customer has been notified", "Hold on, the customer has been notified. A five minute timer is running."),
        },
        "gloss": "Hold on, customer has been notified",
        "buttons": ["Waiting ✓", "Problem"],
        "status": "Customer notified · 5 min timer",
        "ops": "CUSTOMER SMS “Your rider is waiting at the drop point” · OPS: 5-min wait timer on {sit}",
    },
    "call_customer": {
        "text": {
            "hinglish": ("Customer ko call kar rahe hain", "Customer ko call kar rahe hain, line pe rahiye."),
            "hindi": ("कस्टमर को कॉल कर रहे हैं", "कस्टमर को कॉल कर रहे हैं, लाइन पर रहिए।"),
            "english": ("Calling the customer for you", "Calling the customer for you, please stay on the line."),
        },
        "gloss": "Calling the customer for you",
        "buttons": ["Connecting…", "Cancel"],
        "status": "Masked call placed",
        "ops": "OPS: masked call rider ↔ customer · log “{sit} → call customer”",
    },
    "contact_support": {
        "text": {
            "hinglish": ("Support se jod rahe hain", "Support se jod rahe hain. Aapka voice note bhej diya hai."),
            "hindi": ("सपोर्ट से जोड़ रहे हैं", "सपोर्ट से जोड़ रहे हैं। आपका वॉइस नोट भेज दिया है।"),
            "english": ("Connecting you to support", "Connecting you to support. Your voice note has been sent."),
        },
        "gloss": "Connecting you to support",
        "buttons": ["Talk now", "Later"],
        "status": "Ticket raised with your voice note",
        "ops": "OPS TICKET: {sit} → talk to support · voice note attached",
    },
    "request_reassign": {
        "text": {
            "hinglish": ("Order dusre rider ko de rahe hain", "Order dusre rider ko de rahe hain. Confirm karne ke liye tap karein."),
            "hindi": ("ऑर्डर दूसरे राइडर को दे रहे हैं", "ऑर्डर दूसरे राइडर को दे रहे हैं। पक्का करने के लिए टैप करें।"),
            "english": ("Reassigning the order to another rider", "Reassigning the order to another rider. Tap to confirm."),
        },
        "gloss": "Reassigning the order to another rider",
        "buttons": ["Confirm", "Keep order"],
        "status": "Reassign request pending your tap",
        "ops": "OPS: reassign request · {sit}",
    },
    "return_to_restaurant": {
        "text": {
            "hinglish": ("Restaurant wapas jaiye", "Restaurant wapas jaiye, unko bata diya hai."),
            "hindi": ("रेस्टोरेंट वापस जाइए", "रेस्टोरेंट वापस जाइए, उनको बता दिया है।"),
            "english": ("Head back to the restaurant", "Head back to the restaurant, they have been told."),
        },
        "gloss": "Head back to the restaurant",
        "buttons": ["Navigate", "Problem"],
        "status": "Restaurant informed",
        "ops": "RESTAURANT + OPS: rider returning · {sit}",
    },
    "choose": {
        "text": {
            "hinglish": ("Kya karna hai? Ek chuniye", "Mujhe pakka nahi pata. Kya karna hai, ek chuniye."),
            "hindi": ("क्या करना है? एक चुनिए", "मुझे पक्का नहीं पता। क्या करना है, एक चुनिए।"),
            "english": ("Not sure — pick what to do", "I'm not sure. Please pick what to do."),
        },
        "gloss": "Not sure — pick what to do",
        "buttons": [],  # filled from the card's alternatives
        "status": "Low confidence · rider decides",
        "ops": "OPS: nothing sent until the rider taps · {sit}",
    },
    "checkin": {
        "text": {
            "hinglish": ("Aap theek ho?", "Aap theek ho? Tees second mein jawab nahi aaya to support ko bhejenge."),
            "hindi": ("आप ठीक हो?", "आप ठीक हो? तीस सेकंड में जवाब नहीं आया तो सपोर्ट को भेजेंगे।"),
            "english": ("Are you safe?", "Are you safe? If you don't reply in thirty seconds, we'll alert support."),
        },
        "gloss": "Are you safe?",
        "buttons": ["I'm safe", "Need help"],
        "status": "No reply in 30 s → escalate to support",
        "ops": "OPS: safety watch flag · auto-escalates to support after 30 s",
    },
    "sos": {
        "text": {
            "hinglish": ("Madad aa rahi hai", "Madad aa rahi hai. Aapki location support aur emergency contact ko bhej di hai."),
            "hindi": ("मदद आ रही है", "मदद आ रही है। आपकी लोकेशन सपोर्ट और इमरजेंसी कॉन्टैक्ट को भेज दी है।"),
            "english": ("Help is on the way", "Help is on the way. Your location has been shared with support and your emergency contact."),
        },
        "gloss": "Help is on the way",
        "buttons": ["Cancel (10 s)"],
        "status": "Live location shared with support + emergency contact",
        "ops": "SUPPORT + EMERGENCY CONTACT: live location, voice note, {sit}",
    },
}

ACTION_BTN = {"wait": "Wait", "call_customer": "Call customer", "contact_support": "Support",
              "request_reassign": "Reassign", "return_to_restaurant": "Go back", "sos": "SOS"}


def rider_view(card: dict, answers: dict, sit_label: str) -> dict:
    kind = card.get("kind")
    key = kind if kind in ("sos", "checkin", "choose") else (card.get("primary") or answers["action"]["choice"])
    c = COPY.get(key, COPY["contact_support"])
    lang = answers.get("language", {}).get("choice", "english")
    lang = lang if lang in c["text"] else "english"
    line, spoken = c["text"][lang]
    buttons = c["buttons"] or [ACTION_BTN.get(a, a) for a in card.get("alternatives", [])][:2]
    then = None
    if kind == "checkin" and card.get("primary"):
        then = COPY.get(card["primary"], {}).get("gloss")
    return {"key": key, "kind": kind, "lang": lang, "line": line, "gloss": c["gloss"], "spoken": spoken,
            "buttons": buttons, "status": c["status"], "ops": c["ops"].format(sit=sit_label), "then": then}

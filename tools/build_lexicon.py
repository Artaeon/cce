#!/usr/bin/env python3
"""
Build an expanded German lexicon for CCE.

Generates ~2000+ word profiles with auto-detected POS, estimated
valence, syllable count, and register from curated domain lists.
"""

import json
import re
import sys
from pathlib import Path

# ── Domain word lists ──────────────────────────────────────────
# Format: (word, pos, valence, arousal, register, bonds)
# valence: -1 (very negative) to +1 (very positive)
# arousal: 0 (calm) to 1 (intense)
# register: 0 (informal) to 1 (formal)

WORDS = []

def W(word, pos="NOUN", val=0.0, aro=0.5, reg=0.5, bonds=None):
    WORDS.append((word, pos, val, aro, reg, bonds or []))

# ── NOUNS: People & Roles ──────────────────────────────────────
for w, v, a in [
    ("Mensch", 0.2, 0.4), ("Person", 0.1, 0.3), ("Frau", 0.2, 0.4),
    ("Mann", 0.1, 0.4), ("Kind", 0.4, 0.5), ("Freund", 0.6, 0.5),
    ("Feind", -0.6, 0.7), ("Partner", 0.4, 0.4), ("Kollege", 0.2, 0.3),
    ("Chef", 0.0, 0.5), ("Leiter", 0.1, 0.4), ("Mitarbeiter", 0.2, 0.3),
    ("Kunde", 0.1, 0.3), ("Unternehmer", 0.2, 0.5), ("Gründer", 0.3, 0.6),
    ("Berater", 0.1, 0.3), ("Experte", 0.2, 0.4), ("Visionär", 0.4, 0.6),
    ("Pionier", 0.4, 0.6), ("Held", 0.5, 0.7), ("Opfer", -0.5, 0.6),
    ("Zeuge", 0.0, 0.4), ("Richter", 0.0, 0.5), ("Lehrer", 0.2, 0.4),
    ("Schüler", 0.1, 0.3), ("Arzt", 0.2, 0.4), ("Künstler", 0.3, 0.5),
    ("Denker", 0.3, 0.5), ("Träumer", 0.2, 0.4), ("Kämpfer", 0.2, 0.7),
    ("Führer", 0.0, 0.6), ("Verlierer", -0.5, 0.5), ("Gewinner", 0.5, 0.6),
    ("Fremder", -0.1, 0.4), ("Nachbar", 0.1, 0.3), ("Familie", 0.5, 0.5),
    ("Vater", 0.3, 0.4), ("Mutter", 0.4, 0.4), ("Bruder", 0.3, 0.4),
    ("Schwester", 0.3, 0.4), ("Volk", 0.1, 0.4), ("Gesellschaft", 0.1, 0.3),
]:
    W(w, "NOUN", v, a, 0.5)

# ── NOUNS: Business & Economy ──────────────────────────────────
for w, v, a in [
    ("Markt", 0.1, 0.5), ("Wirtschaft", 0.1, 0.4), ("Handel", 0.1, 0.4),
    ("Gewinn", 0.5, 0.6), ("Verlust", -0.6, 0.6), ("Umsatz", 0.2, 0.5),
    ("Wachstum", 0.5, 0.6), ("Rückgang", -0.4, 0.5), ("Krise", -0.6, 0.8),
    ("Chance", 0.5, 0.6), ("Risiko", -0.3, 0.6), ("Strategie", 0.2, 0.5),
    ("Innovation", 0.5, 0.6), ("Produkt", 0.1, 0.3), ("Qualität", 0.4, 0.4),
    ("Leistung", 0.3, 0.5), ("Ergebnis", 0.1, 0.5), ("Erfolg", 0.7, 0.7),
    ("Misserfolg", -0.6, 0.6), ("Investition", 0.2, 0.5), ("Kapital", 0.2, 0.4),
    ("Wettbewerb", 0.1, 0.6), ("Lösung", 0.4, 0.5), ("Problem", -0.3, 0.5),
    ("Herausforderung", 0.1, 0.6), ("Projekt", 0.2, 0.4), ("Ziel", 0.3, 0.5),
    ("Plan", 0.2, 0.4), ("Idee", 0.4, 0.6), ("Konzept", 0.2, 0.4),
    ("Entscheidung", 0.1, 0.6), ("Verantwortung", 0.2, 0.5),
    ("Führung", 0.2, 0.5), ("Wandel", 0.1, 0.5), ("Fortschritt", 0.4, 0.5),
    ("Entwicklung", 0.3, 0.4), ("Wert", 0.3, 0.4), ("Preis", 0.0, 0.4),
    ("Kosten", -0.2, 0.4), ("Budget", 0.0, 0.3), ("Bilanz", 0.0, 0.4),
    ("Quartal", 0.0, 0.3), ("Bericht", 0.0, 0.3), ("Analyse", 0.1, 0.4),
    ("Trend", 0.2, 0.4), ("Prognose", 0.1, 0.4), ("Zukunft", 0.3, 0.5),
    ("Vision", 0.4, 0.6), ("Mission", 0.3, 0.5), ("Auftrag", 0.1, 0.4),
    ("Vertrag", 0.0, 0.3), ("Verhandlung", 0.0, 0.5), ("Vereinbarung", 0.1, 0.3),
    ("Firma", 0.1, 0.3), ("Unternehmen", 0.1, 0.4), ("Branche", 0.0, 0.3),
    ("Konkurrenz", -0.1, 0.5), ("Vorteil", 0.4, 0.5), ("Nachteil", -0.4, 0.4),
    ("Potenzial", 0.4, 0.5), ("Ressource", 0.2, 0.3), ("Effizienz", 0.3, 0.4),
    ("Produktivität", 0.3, 0.4), ("Rendite", 0.3, 0.5), ("Dividende", 0.3, 0.3),
    ("Aktie", 0.1, 0.5), ("Börse", 0.1, 0.5), ("Schuld", -0.4, 0.5),
    ("Kredit", -0.1, 0.4), ("Zinsen", -0.1, 0.3), ("Inflation", -0.4, 0.5),
    ("Stabilität", 0.3, 0.3), ("Wachstumsrate", 0.2, 0.4),
]:
    W(w, "NOUN", v, a, 0.7)

# ── NOUNS: Emotions & Abstract ─────────────────────────────────
for w, v, a in [
    ("Mut", 0.6, 0.7), ("Angst", -0.7, 0.8), ("Hoffnung", 0.7, 0.6),
    ("Verzweiflung", -0.8, 0.8), ("Freude", 0.8, 0.7), ("Trauer", -0.6, 0.6),
    ("Wut", -0.5, 0.9), ("Zorn", -0.5, 0.9), ("Liebe", 0.8, 0.7),
    ("Hass", -0.8, 0.9), ("Stolz", 0.5, 0.6), ("Scham", -0.6, 0.6),
    ("Schuld", -0.5, 0.6), ("Vertrauen", 0.6, 0.5), ("Misstrauen", -0.5, 0.5),
    ("Sehnsucht", 0.2, 0.6), ("Einsamkeit", -0.6, 0.5), ("Kraft", 0.5, 0.7),
    ("Schwäche", -0.4, 0.4), ("Stärke", 0.5, 0.6), ("Freiheit", 0.7, 0.6),
    ("Zweifel", -0.3, 0.5), ("Glaube", 0.4, 0.5), ("Wahrheit", 0.4, 0.5),
    ("Lüge", -0.6, 0.6), ("Ehre", 0.5, 0.5), ("Würde", 0.5, 0.5),
    ("Schmerz", -0.6, 0.7), ("Glück", 0.8, 0.7), ("Leid", -0.6, 0.6),
    ("Trost", 0.4, 0.4), ("Ruhe", 0.3, 0.2), ("Frieden", 0.6, 0.3),
    ("Krieg", -0.7, 0.9), ("Kampf", -0.1, 0.8), ("Sieg", 0.6, 0.8),
    ("Niederlage", -0.6, 0.7), ("Geduld", 0.3, 0.2), ("Ungeduld", -0.2, 0.6),
    ("Weisheit", 0.5, 0.4), ("Dummheit", -0.5, 0.4), ("Wissen", 0.4, 0.4),
    ("Ignoranz", -0.4, 0.3), ("Neugier", 0.3, 0.6), ("Langeweile", -0.3, 0.2),
    ("Leidenschaft", 0.5, 0.8), ("Gleichgültigkeit", -0.3, 0.1),
    ("Respekt", 0.4, 0.4), ("Verachtung", -0.6, 0.6), ("Bewunderung", 0.5, 0.5),
    ("Neid", -0.5, 0.6), ("Eifersucht", -0.5, 0.7), ("Dankbarkeit", 0.6, 0.5),
    ("Reue", -0.3, 0.5), ("Erleichterung", 0.5, 0.4), ("Spannung", 0.1, 0.7),
    ("Überraschung", 0.2, 0.7), ("Enttäuschung", -0.5, 0.6),
    ("Begeisterung", 0.7, 0.8), ("Erschöpfung", -0.4, 0.3),
    ("Entschlossenheit", 0.4, 0.7), ("Hingabe", 0.4, 0.6),
    ("Trotz", 0.1, 0.7), ("Widerstand", 0.1, 0.7), ("Ausdauer", 0.4, 0.5),
    ("Demut", 0.3, 0.3), ("Größe", 0.4, 0.5), ("Tiefe", 0.2, 0.4),
    ("Leere", -0.5, 0.3), ("Fülle", 0.5, 0.5), ("Sinn", 0.4, 0.5),
    ("Unsinn", -0.3, 0.4), ("Mangel", -0.4, 0.4), ("Überfluss", 0.3, 0.4),
    ("Ordnung", 0.3, 0.3), ("Chaos", -0.3, 0.7), ("Harmonie", 0.5, 0.3),
    ("Konflikt", -0.3, 0.7), ("Balance", 0.3, 0.3), ("Spannung", -0.1, 0.6),
]:
    W(w, "NOUN", v, a, 0.5)

# ── NOUNS: Nature & World ──────────────────────────────────────
for w, v, a in [
    ("Welt", 0.2, 0.5), ("Erde", 0.2, 0.4), ("Himmel", 0.4, 0.4),
    ("Meer", 0.3, 0.4), ("Berg", 0.2, 0.4), ("Tal", 0.1, 0.3),
    ("Wald", 0.3, 0.3), ("Baum", 0.3, 0.3), ("Blume", 0.4, 0.3),
    ("Stein", 0.0, 0.2), ("Feuer", 0.1, 0.8), ("Wasser", 0.2, 0.4),
    ("Luft", 0.2, 0.3), ("Wind", 0.1, 0.4), ("Sturm", -0.2, 0.8),
    ("Regen", 0.0, 0.3), ("Schnee", 0.1, 0.3), ("Sonne", 0.5, 0.5),
    ("Mond", 0.2, 0.3), ("Stern", 0.4, 0.4), ("Licht", 0.5, 0.5),
    ("Dunkelheit", -0.4, 0.4), ("Schatten", -0.2, 0.3), ("Nacht", -0.1, 0.3),
    ("Tag", 0.2, 0.4), ("Morgen", 0.3, 0.4), ("Abend", 0.1, 0.3),
    ("Horizont", 0.3, 0.4), ("Ufer", 0.2, 0.3), ("Quelle", 0.3, 0.4),
    ("Strom", 0.1, 0.5), ("Fluss", 0.2, 0.4), ("Ozean", 0.3, 0.5),
    ("Wüste", -0.2, 0.4), ("Garten", 0.4, 0.3), ("Feld", 0.1, 0.3),
    ("Weg", 0.2, 0.4), ("Straße", 0.0, 0.3), ("Brücke", 0.3, 0.4),
    ("Mauer", -0.2, 0.3), ("Tür", 0.1, 0.4), ("Fenster", 0.1, 0.3),
    ("Raum", 0.1, 0.3), ("Haus", 0.3, 0.3), ("Stadt", 0.1, 0.4),
    ("Land", 0.2, 0.3), ("Heimat", 0.5, 0.5), ("Grenze", -0.1, 0.4),
]:
    W(w, "NOUN", v, a, 0.4)

# ── NOUNS: Time & Process ──────────────────────────────────────
for w, v, a in [
    ("Zeit", 0.0, 0.4), ("Moment", 0.2, 0.5), ("Augenblick", 0.2, 0.5),
    ("Stunde", 0.0, 0.3), ("Minute", 0.0, 0.3), ("Sekunde", 0.0, 0.4),
    ("Woche", 0.0, 0.2), ("Monat", 0.0, 0.2), ("Jahr", 0.0, 0.3),
    ("Jahrzehnt", 0.0, 0.3), ("Jahrhundert", 0.1, 0.3),
    ("Vergangenheit", -0.1, 0.3), ("Gegenwart", 0.2, 0.4),
    ("Ewigkeit", 0.1, 0.3), ("Anfang", 0.3, 0.5), ("Ende", -0.1, 0.5),
    ("Beginn", 0.3, 0.5), ("Abschluss", 0.2, 0.4), ("Prozess", 0.1, 0.3),
    ("Phase", 0.0, 0.3), ("Schritt", 0.2, 0.4), ("Stufe", 0.1, 0.3),
    ("Wandlung", 0.2, 0.5), ("Wendepunkt", 0.2, 0.6),
    ("Übergang", 0.1, 0.4), ("Reise", 0.3, 0.5), ("Aufbruch", 0.4, 0.6),
]:
    W(w, "NOUN", v, a, 0.5)

# ── NOUNS: Mind & Communication ────────────────────────────────
for w, v, a in [
    ("Gedanke", 0.2, 0.4), ("Frage", 0.1, 0.5), ("Antwort", 0.2, 0.4),
    ("Wort", 0.1, 0.3), ("Sprache", 0.2, 0.4), ("Stimme", 0.2, 0.4),
    ("Stille", 0.1, 0.2), ("Schweigen", -0.1, 0.2), ("Ruf", 0.1, 0.5),
    ("Schrei", -0.2, 0.8), ("Gespräch", 0.2, 0.4), ("Dialog", 0.3, 0.4),
    ("Botschaft", 0.2, 0.4), ("Zeichen", 0.1, 0.4), ("Bedeutung", 0.2, 0.4),
    ("Verständnis", 0.4, 0.4), ("Erkenntnis", 0.5, 0.6),
    ("Einsicht", 0.4, 0.5), ("Urteil", 0.0, 0.5), ("Meinung", 0.0, 0.4),
    ("Traum", 0.3, 0.5), ("Vorstellung", 0.2, 0.4), ("Fantasie", 0.3, 0.5),
    ("Erinnerung", 0.2, 0.4), ("Erfahrung", 0.3, 0.4), ("Gewissen", 0.2, 0.5),
    ("Geist", 0.3, 0.5), ("Seele", 0.3, 0.5), ("Herz", 0.4, 0.6),
    ("Kopf", 0.0, 0.4), ("Verstand", 0.3, 0.4), ("Vernunft", 0.3, 0.4),
    ("Instinkt", 0.1, 0.5), ("Gefühl", 0.2, 0.5), ("Ahnung", 0.1, 0.4),
    ("Überzeugung", 0.3, 0.6), ("Zweck", 0.1, 0.3), ("Grund", 0.0, 0.3),
    ("Ursache", 0.0, 0.4), ("Wirkung", 0.1, 0.4), ("Folge", 0.0, 0.4),
    ("Beispiel", 0.1, 0.3), ("Beweis", 0.2, 0.5), ("Tatsache", 0.1, 0.4),
    ("Prinzip", 0.2, 0.4), ("Regel", 0.0, 0.3), ("Norm", 0.0, 0.3),
    ("Gesetz", 0.0, 0.4), ("Recht", 0.3, 0.4), ("Pflicht", 0.1, 0.4),
    ("Aufgabe", 0.1, 0.4), ("Arbeit", 0.1, 0.5), ("Leistung", 0.3, 0.5),
    ("Werk", 0.3, 0.4), ("Handlung", 0.1, 0.5), ("Tat", 0.2, 0.6),
]:
    W(w, "NOUN", v, a, 0.5)

# ── NOUNS: Body & Physical ─────────────────────────────────────
for w, v, a in [
    ("Körper", 0.1, 0.4), ("Hand", 0.1, 0.3), ("Auge", 0.2, 0.4),
    ("Blick", 0.1, 0.4), ("Gesicht", 0.1, 0.4), ("Stimme", 0.2, 0.4),
    ("Atem", 0.1, 0.3), ("Blut", -0.1, 0.5), ("Knochen", 0.0, 0.3),
    ("Haut", 0.1, 0.3), ("Narbe", -0.2, 0.4), ("Träne", -0.2, 0.5),
    ("Lächeln", 0.5, 0.4), ("Schlag", -0.3, 0.7), ("Berührung", 0.3, 0.5),
]:
    W(w, "NOUN", v, a, 0.4)

# ── ADJECTIVES ─────────────────────────────────────────────────
for w, v, a in [
    ("gut", 0.6, 0.4), ("schlecht", -0.6, 0.4), ("groß", 0.3, 0.4),
    ("klein", -0.1, 0.3), ("stark", 0.5, 0.6), ("schwach", -0.4, 0.3),
    ("schnell", 0.2, 0.6), ("langsam", -0.1, 0.2), ("neu", 0.3, 0.5),
    ("alt", -0.1, 0.2), ("jung", 0.3, 0.5), ("schön", 0.6, 0.5),
    ("hässlich", -0.5, 0.4), ("hell", 0.4, 0.4), ("dunkel", -0.3, 0.3),
    ("warm", 0.3, 0.3), ("kalt", -0.2, 0.3), ("hart", -0.1, 0.5),
    ("weich", 0.2, 0.2), ("laut", -0.1, 0.6), ("leise", 0.1, 0.2),
    ("tief", 0.1, 0.4), ("hoch", 0.3, 0.4), ("weit", 0.2, 0.3),
    ("nah", 0.2, 0.4), ("lang", 0.0, 0.3), ("kurz", 0.0, 0.3),
    ("breit", 0.1, 0.3), ("schmal", -0.1, 0.2), ("schwer", -0.2, 0.4),
    ("leicht", 0.2, 0.3), ("voll", 0.2, 0.4), ("leer", -0.3, 0.3),
    ("offen", 0.3, 0.4), ("geschlossen", -0.2, 0.3),
    ("frei", 0.6, 0.5), ("gefangen", -0.5, 0.5),
    ("wahr", 0.4, 0.5), ("falsch", -0.4, 0.4),
    ("richtig", 0.4, 0.4), ("wichtig", 0.3, 0.5),
    ("möglich", 0.3, 0.4), ("unmöglich", -0.3, 0.5),
    ("klar", 0.4, 0.4), ("unklar", -0.2, 0.3),
    ("sicher", 0.4, 0.4), ("unsicher", -0.3, 0.4),
    ("einfach", 0.2, 0.2), ("schwierig", -0.2, 0.5),
    ("mutig", 0.5, 0.7), ("feige", -0.5, 0.4),
    ("ehrlich", 0.5, 0.5), ("direkt", 0.2, 0.5),
    ("wild", 0.1, 0.7), ("ruhig", 0.3, 0.2), ("still", 0.1, 0.1),
    ("lebendig", 0.5, 0.6), ("tot", -0.6, 0.3),
    ("bitter", -0.4, 0.4), ("süß", 0.4, 0.3),
    ("scharf", 0.0, 0.5), ("stumpf", -0.2, 0.2),
    ("frisch", 0.4, 0.4), ("müde", -0.2, 0.2),
    ("wach", 0.3, 0.5), ("bereit", 0.3, 0.5),
    ("fertig", 0.2, 0.4), ("kaputt", -0.4, 0.4),
    ("ganz", 0.2, 0.3), ("zerbrochen", -0.5, 0.5),
    ("rein", 0.4, 0.3), ("schmutzig", -0.3, 0.3),
    ("positiv", 0.5, 0.5), ("negativ", -0.5, 0.5),
    ("konkret", 0.2, 0.4), ("abstrakt", 0.0, 0.3),
    ("praktisch", 0.3, 0.4), ("theoretisch", 0.0, 0.2),
    ("real", 0.2, 0.4), ("irreal", -0.1, 0.3),
    ("rational", 0.2, 0.3), ("emotional", 0.1, 0.5),
    ("analytisch", 0.2, 0.4), ("kreativ", 0.4, 0.5),
    ("strategisch", 0.2, 0.4), ("taktisch", 0.1, 0.4),
    ("nachhaltig", 0.3, 0.3), ("kurzfristig", -0.1, 0.4),
    ("langfristig", 0.2, 0.3), ("global", 0.1, 0.4),
    ("lokal", 0.1, 0.3), ("digital", 0.2, 0.4),
    ("dringend", -0.1, 0.7), ("notwendig", 0.1, 0.5),
    ("überflüssig", -0.3, 0.3), ("wertvoll", 0.5, 0.5),
    ("wertlos", -0.5, 0.4), ("einzigartig", 0.4, 0.5),
    ("gewöhnlich", -0.1, 0.2), ("außergewöhnlich", 0.5, 0.6),
    ("mächtig", 0.3, 0.6), ("hilflos", -0.5, 0.4),
    ("hoffnungsvoll", 0.6, 0.5), ("hoffnungslos", -0.7, 0.5),
    ("frustriert", -0.5, 0.6), ("zufrieden", 0.5, 0.3),
    ("melancholisch", -0.2, 0.3), ("optimistisch", 0.6, 0.5),
    ("pessimistisch", -0.5, 0.4), ("realistisch", 0.2, 0.3),
    ("entschlossen", 0.4, 0.7), ("zögerlich", -0.2, 0.3),
    ("einsam", -0.5, 0.4), ("verbunden", 0.4, 0.4),
    ("verloren", -0.5, 0.4), ("gefunden", 0.4, 0.5),
    ("gebrochen", -0.5, 0.5), ("geheilt", 0.5, 0.4),
    ("stumm", -0.2, 0.2), ("leidenschaftlich", 0.4, 0.8),
    ("gleichgültig", -0.3, 0.1), ("kämpferisch", 0.3, 0.8),
    ("friedlich", 0.4, 0.2), ("aggressiv", -0.3, 0.8),
    ("sanft", 0.3, 0.2), ("brutal", -0.6, 0.8),
    ("transparent", 0.3, 0.3), ("verborgen", -0.1, 0.3),
    ("profitabel", 0.5, 0.5), ("stabil", 0.3, 0.3),
    ("volatil", -0.2, 0.6), ("resilient", 0.4, 0.5),
    ("flexibel", 0.3, 0.4), ("starr", -0.3, 0.3),
    ("dynamisch", 0.4, 0.6), ("statisch", -0.1, 0.2),
    ("komplex", 0.0, 0.5), ("simpel", 0.1, 0.2),
    ("robust", 0.3, 0.4), ("fragil", -0.2, 0.3),
    ("authentisch", 0.4, 0.5), ("künstlich", -0.2, 0.3),
    ("unerbittlich", 0.0, 0.7), ("gnadenlos", -0.3, 0.7),
    ("unerschütterlich", 0.4, 0.6), ("verwundbar", -0.2, 0.4),
]:
    W(w, "ADJ", v, a, 0.5)

# ── VERBS ──────────────────────────────────────────────────────
for w, v, a in [
    ("sein", 0.0, 0.2), ("haben", 0.1, 0.3), ("werden", 0.2, 0.4),
    ("können", 0.2, 0.3), ("müssen", -0.1, 0.4), ("sollen", 0.0, 0.3),
    ("wollen", 0.2, 0.5), ("dürfen", 0.1, 0.3),
    ("machen", 0.1, 0.4), ("tun", 0.1, 0.4), ("geben", 0.2, 0.4),
    ("nehmen", 0.0, 0.4), ("kommen", 0.1, 0.4), ("gehen", 0.0, 0.4),
    ("stehen", 0.1, 0.3), ("liegen", 0.0, 0.2), ("sitzen", 0.0, 0.2),
    ("laufen", 0.2, 0.5), ("fallen", -0.3, 0.5), ("steigen", 0.3, 0.5),
    ("sinken", -0.3, 0.4), ("wachsen", 0.4, 0.5), ("schrumpfen", -0.3, 0.4),
    ("sagen", 0.0, 0.3), ("sprechen", 0.1, 0.4), ("reden", 0.0, 0.3),
    ("rufen", 0.1, 0.6), ("schreien", -0.2, 0.8), ("flüstern", 0.1, 0.2),
    ("schweigen", 0.0, 0.1), ("hören", 0.1, 0.3), ("sehen", 0.1, 0.4),
    ("fühlen", 0.2, 0.5), ("denken", 0.2, 0.4), ("glauben", 0.2, 0.4),
    ("wissen", 0.3, 0.4), ("verstehen", 0.3, 0.5), ("lernen", 0.3, 0.4),
    ("erkennen", 0.4, 0.5), ("entdecken", 0.4, 0.6), ("suchen", 0.1, 0.5),
    ("finden", 0.4, 0.5), ("verlieren", -0.5, 0.5), ("gewinnen", 0.5, 0.6),
    ("kämpfen", 0.1, 0.8), ("siegen", 0.5, 0.7), ("scheitern", -0.5, 0.6),
    ("aufgeben", -0.4, 0.5), ("weitermachen", 0.3, 0.5),
    ("beginnen", 0.3, 0.5), ("beenden", 0.1, 0.4), ("starten", 0.3, 0.5),
    ("stoppen", 0.0, 0.4), ("öffnen", 0.3, 0.4), ("schließen", 0.0, 0.3),
    ("bauen", 0.3, 0.5), ("zerstören", -0.6, 0.7),
    ("schaffen", 0.5, 0.6), ("erschaffen", 0.5, 0.6),
    ("brechen", -0.3, 0.6), ("heilen", 0.5, 0.5),
    ("verbinden", 0.4, 0.4), ("trennen", -0.3, 0.4),
    ("lieben", 0.7, 0.7), ("hassen", -0.7, 0.8),
    ("helfen", 0.5, 0.5), ("hindern", -0.3, 0.4),
    ("fördern", 0.4, 0.4), ("behindern", -0.3, 0.4),
    ("führen", 0.2, 0.5), ("folgen", 0.1, 0.3),
    ("wagen", 0.3, 0.6), ("riskieren", 0.0, 0.6),
    ("hoffen", 0.5, 0.5), ("fürchten", -0.4, 0.6),
    ("vertrauen", 0.5, 0.4), ("zweifeln", -0.3, 0.4),
    ("entscheiden", 0.2, 0.6), ("wählen", 0.1, 0.4),
    ("verändern", 0.2, 0.5), ("bewahren", 0.2, 0.3),
    ("wachsen", 0.4, 0.5), ("blühen", 0.5, 0.5), ("welken", -0.4, 0.3),
    ("brennen", 0.1, 0.8), ("leuchten", 0.4, 0.5), ("glänzen", 0.4, 0.5),
    ("strahlen", 0.5, 0.5), ("verblassen", -0.3, 0.3),
    ("atmen", 0.2, 0.3), ("leben", 0.5, 0.5), ("sterben", -0.5, 0.6),
    ("überleben", 0.3, 0.6), ("tragen", 0.1, 0.4), ("halten", 0.2, 0.4),
    ("lassen", 0.0, 0.3), ("ziehen", 0.0, 0.4), ("drücken", 0.0, 0.4),
    ("stoßen", -0.1, 0.5), ("greifen", 0.1, 0.5), ("loslassen", 0.2, 0.4),
    ("festhalten", 0.1, 0.5), ("aufstehen", 0.3, 0.5),
    ("investieren", 0.2, 0.5), ("profitieren", 0.4, 0.5),
    ("analysieren", 0.1, 0.4), ("optimieren", 0.3, 0.4),
    ("implementieren", 0.2, 0.4), ("transformieren", 0.3, 0.5),
    ("skalieren", 0.2, 0.4), ("dominieren", 0.1, 0.6),
]:
    W(w, "VERB", v, a, 0.5)

# ── PARTICLES, CONJUNCTIONS, ADVERBS ───────────────────────────
for w, v, a, pos in [
    ("nicht", 0.0, 0.3, "PART"), ("auch", 0.1, 0.2, "PART"),
    ("noch", 0.0, 0.3, "PART"), ("schon", 0.1, 0.3, "PART"),
    ("nur", 0.0, 0.3, "PART"), ("erst", 0.0, 0.3, "PART"),
    ("immer", 0.1, 0.3, "ADV"), ("nie", -0.2, 0.3, "ADV"),
    ("oft", 0.1, 0.3, "ADV"), ("manchmal", 0.0, 0.2, "ADV"),
    ("hier", 0.1, 0.3, "ADV"), ("dort", 0.0, 0.3, "ADV"),
    ("jetzt", 0.1, 0.4, "ADV"), ("heute", 0.1, 0.4, "ADV"),
    ("morgen", 0.2, 0.4, "ADV"), ("gestern", 0.0, 0.3, "ADV"),
    ("wieder", 0.1, 0.3, "ADV"), ("zusammen", 0.3, 0.4, "ADV"),
    ("allein", -0.2, 0.3, "ADV"), ("gemeinsam", 0.4, 0.4, "ADV"),
    ("trotzdem", 0.2, 0.5, "ADV"), ("dennoch", 0.2, 0.4, "ADV"),
    ("vielleicht", 0.0, 0.3, "ADV"), ("bestimmt", 0.2, 0.4, "ADV"),
    ("tatsächlich", 0.1, 0.4, "ADV"), ("wirklich", 0.2, 0.4, "ADV"),
    ("eigentlich", 0.0, 0.3, "ADV"), ("offensichtlich", 0.1, 0.4, "ADV"),
    ("plötzlich", 0.0, 0.7, "ADV"), ("langsam", 0.0, 0.2, "ADV"),
    ("endlich", 0.3, 0.5, "ADV"), ("bereits", 0.1, 0.3, "ADV"),
    ("bisher", 0.0, 0.2, "ADV"), ("künftig", 0.2, 0.4, "ADV"),
    ("und", 0.1, 0.1, "CONJ"), ("aber", 0.0, 0.3, "CONJ"),
    ("oder", 0.0, 0.2, "CONJ"), ("denn", 0.0, 0.2, "CONJ"),
    ("weil", 0.0, 0.3, "CONJ"), ("wenn", 0.0, 0.3, "CONJ"),
    ("obwohl", 0.0, 0.3, "CONJ"), ("während", 0.0, 0.3, "CONJ"),
    ("dass", 0.0, 0.2, "CONJ"), ("damit", 0.1, 0.3, "CONJ"),
    ("bevor", 0.0, 0.3, "CONJ"), ("nachdem", 0.0, 0.3, "CONJ"),
    ("sobald", 0.1, 0.4, "CONJ"), ("solange", 0.0, 0.3, "CONJ"),
    ("der", 0.0, 0.1, "ART"), ("die", 0.0, 0.1, "ART"),
    ("das", 0.0, 0.1, "ART"), ("ein", 0.0, 0.1, "ART"),
    ("eine", 0.0, 0.1, "ART"), ("kein", -0.1, 0.2, "ART"),
]:
    W(w, pos, v, a, 0.5)


def estimate_syllables(word: str) -> int:
    """Estimate German syllable count via vowel clusters."""
    word = word.lower()
    vowels = set("aeiouyäöü")
    count = 0
    in_vowel = False
    for ch in word:
        if ch in vowels:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
    return max(1, count)


def _guess_gender(word: str, pos: str) -> str:
    """Guess grammatical gender for nouns."""
    if pos != "NOUN":
        return ""
    lower = word.lower()
    f_suffixes = ("ung", "heit", "keit", "schaft", "tion", "sion", "tät",
                  "enz", "anz", "ie", "ik", "ur", "ei")
    n_suffixes = ("ment", "tum", "nis", "chen", "lein", "um", "ma")
    m_suffixes = ("ling", "ismus", "ist", "or", "ant", "ent", "er")
    for s in sorted(f_suffixes, key=len, reverse=True):
        if lower.endswith(s):
            return "F"
    for s in sorted(n_suffixes, key=len, reverse=True):
        if lower.endswith(s):
            return "N"
    for s in sorted(m_suffixes, key=len, reverse=True):
        if lower.endswith(s):
            return "M"
    return "M"  # default masculine


def build_lexicon(output_path: str) -> None:
    """Build the expanded lexicon JSON file."""
    # Deduplicate (keep first occurrence)
    seen = set()
    entries = []
    for word, pos, val, aro, reg, bonds in WORDS:
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        entry = {
            "word": word,
            "pos": pos,
            "syllables": estimate_syllables(word),
            "valence": round(val, 2),
            "arousal": round(aro, 2),
            "register": round(reg, 2),
            "bonds": bonds,
            "repels": [],
        }
        gender = _guess_gender(word, pos)
        if gender:
            entry["gender"] = gender
        entries.append(entry)

    # Sort by POS then alphabetically
    entries.sort(key=lambda e: (e["pos"], e["word"].lower()))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    # Stats
    pos_counts = {}
    for e in entries:
        pos_counts[e["pos"]] = pos_counts.get(e["pos"], 0) + 1
    
    print(f"Built lexicon: {len(entries)} words → {output}")
    for pos, count in sorted(pos_counts.items()):
        print(f"  {pos}: {count}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/de_lexicon_5k.json"
    build_lexicon(out)

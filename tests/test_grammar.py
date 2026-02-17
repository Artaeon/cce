"""Tests for the German grammar engine (grammar.py)."""

import pytest

from cce.grammar import (
    Case,
    Clause,
    Gender,
    NounPhrase,
    conjugate,
    decline_adjective,
    get_article,
    guess_gender,
)


class TestArticles:
    """Test article declension."""

    def test_definite_nominative(self):
        assert get_article(Gender.M, Case.NOM, "def") == "der"
        assert get_article(Gender.F, Case.NOM, "def") == "die"
        assert get_article(Gender.N, Case.NOM, "def") == "das"

    def test_definite_accusative(self):
        assert get_article(Gender.M, Case.ACC, "def") == "den"
        assert get_article(Gender.F, Case.ACC, "def") == "die"
        assert get_article(Gender.N, Case.ACC, "def") == "das"

    def test_definite_dative(self):
        assert get_article(Gender.M, Case.DAT, "def") == "dem"
        assert get_article(Gender.F, Case.DAT, "def") == "der"
        assert get_article(Gender.N, Case.DAT, "def") == "dem"

    def test_definite_genitive(self):
        assert get_article(Gender.M, Case.GEN, "def") == "des"
        assert get_article(Gender.F, Case.GEN, "def") == "der"
        assert get_article(Gender.N, Case.GEN, "def") == "des"

    def test_indefinite_nominative(self):
        assert get_article(Gender.M, Case.NOM, "indef") == "ein"
        assert get_article(Gender.F, Case.NOM, "indef") == "eine"
        assert get_article(Gender.N, Case.NOM, "indef") == "ein"

    def test_indefinite_accusative(self):
        assert get_article(Gender.M, Case.ACC, "indef") == "einen"
        assert get_article(Gender.F, Case.ACC, "indef") == "eine"

    def test_negative_articles(self):
        assert get_article(Gender.M, Case.NOM, "neg") == "kein"
        assert get_article(Gender.F, Case.NOM, "neg") == "keine"


class TestGenderGuessing:
    """Test suffix-based gender guessing."""

    def test_feminine_suffixes(self):
        assert guess_gender("Hoffnung") == Gender.F
        assert guess_gender("Freiheit") == Gender.F
        assert guess_gender("Einsamkeit") == Gender.F
        assert guess_gender("Wirtschaft") == Gender.F
        assert guess_gender("Innovation") == Gender.F
        assert guess_gender("Qualität") == Gender.F

    def test_neuter_suffixes(self):
        assert guess_gender("Dokument") == Gender.N
        assert guess_gender("Reichtum") == Gender.N
        assert guess_gender("Ergebnis") == Gender.N
        assert guess_gender("Mädchen") == Gender.N

    def test_masculine_suffixes(self):
        assert guess_gender("Lehrling") == Gender.M
        assert guess_gender("Realismus") == Gender.M

    def test_default_masculine(self):
        assert guess_gender("Tisch") == Gender.M


class TestConjugation:
    """Test verb conjugation."""

    def test_irregular_sein(self):
        assert conjugate("sein", "3s") == "ist"

    def test_irregular_haben(self):
        assert conjugate("haben", "3s") == "hat"

    def test_irregular_werden(self):
        assert conjugate("werden", "3s") == "wird"

    def test_irregular_können(self):
        assert conjugate("können", "3s") == "kann"

    def test_irregular_geben(self):
        assert conjugate("geben", "3s") == "gibt"

    def test_regular_machen(self):
        assert conjugate("machen", "3s") == "macht"

    def test_regular_first_person(self):
        assert conjugate("machen", "1s") == "mache"

    def test_regular_plural(self):
        assert conjugate("machen", "1p") == "machen"

    def test_stem_ending_in_t(self):
        # "arbeiten" stem = "arbeit", 3s should be "arbeitet"
        result = conjugate("arbeiten", "3s")
        assert result == "arbeitet"


class TestNounPhrase:
    """Test NounPhrase rendering."""

    def test_simple_noun(self):
        np = NounPhrase(noun="Kraft", gender=Gender.F, article_type="def")
        assert np.render() == "die Kraft"

    def test_noun_with_adjective(self):
        np = NounPhrase(
            noun="Kraft", gender=Gender.F,
            adjective="stark", article_type="def",
        )
        result = np.render()
        assert "die" in result
        assert "Kraft" in result

    def test_accusative(self):
        np = NounPhrase(
            noun="Mann", gender=Gender.M,
            case=Case.ACC, article_type="def",
        )
        assert np.render() == "den Mann"

    def test_no_article(self):
        np = NounPhrase(noun="Mut", gender=Gender.M, article_type="none")
        assert np.render() == "Mut"


class TestClause:
    """Test clause rendering."""

    def test_main_clause_svo(self):
        c = Clause(subject="der Mensch", verb="sucht", object="die Wahrheit")
        result = c.render()
        assert "der Mensch sucht die Wahrheit" == result

    def test_main_clause_predicate(self):
        c = Clause(subject="die Kraft", verb="ist", predicate="stark")
        assert c.render() == "die Kraft ist stark"

    def test_subordinate_v_final(self):
        c = Clause(
            subject="er", verb="sucht",
            object="die Wahrheit",
            clause_type="subordinate", connector="weil",
        )
        result = c.render()
        # V-final: verb should be last
        assert result.endswith("sucht")
        assert result.startswith("weil")

    def test_negation(self):
        c = Clause(subject="er", verb="kommt", negation=True)
        result = c.render()
        assert "nicht" in result

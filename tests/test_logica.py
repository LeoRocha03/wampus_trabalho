"""Testes da lógica do Agente B (CNF, resolução, TELL/ASK) e dos agentes.

Rode com:  python -m unittest discover -s tests -v

sympy é usado SÓ AQUI, para conferir os resultados da nossa implementação
(é permitido nos testes; o núcleo do agente não depende dele). Se sympy não
estiver instalado, esses testes específicos são pulados.
"""

import random
import unittest

from agentes.agente_logico import (B, E, F, M, W, AgenteLogico, BaseConhecimento,
                                   Equivale, Implica, Nao, Ou, P, axiomas_da_casa,
                                   axiomas_iniciais, para_cnf, pl_resolucao,
                                   resolver, sentenca_segura)
from agentes.agente_objetivo import AgenteObjetivo
from agentes.agente_reativo import AgenteReativo
from wumpus import Acao, MundoWumpus, Percepcao, rodar_episodio

try:
    import sympy
    from sympy.logic.inference import satisfiable
except ImportError:  # pragma: no cover
    sympy = None


def lit(s, pos=True):
    return (s, pos)


class TestCNF(unittest.TestCase):
    def test_bicondicional_de_brisa(self):
        cnf = para_cnf(Equivale("B11", Ou("P12", "P21")))
        esperado = {
            frozenset({lit("B11", False), lit("P12"), lit("P21")}),
            frozenset({lit("P12", False), lit("B11")}),
            frozenset({lit("P21", False), lit("B11")}),
        }
        self.assertEqual(cnf, esperado)

    def test_implicacao_e_de_morgan(self):
        self.assertEqual(para_cnf(Implica("A", "B")), {frozenset({lit("A", False), lit("B")})})
        self.assertEqual(para_cnf(Nao(E("A", "B"))),
                         {frozenset({lit("A", False), lit("B", False)})})
        self.assertEqual(para_cnf(Nao(Ou("A", "B"))),
                         {frozenset({lit("A", False)}), frozenset({lit("B", False)})})
        self.assertEqual(para_cnf(Nao(Nao("A"))), {frozenset({lit("A")})})

    def test_distribuicao(self):
        # A ∨ (B ∧ C)  ≡  (A ∨ B) ∧ (A ∨ C)
        self.assertEqual(para_cnf(Ou("A", E("B", "C"))),
                         {frozenset({lit("A"), lit("B")}), frozenset({lit("A"), lit("C")})})

    def test_tautologia_some(self):
        self.assertEqual(para_cnf(Ou("A", Nao("A"))), set())


class TestResolucao(unittest.TestCase):
    def test_resolvente(self):
        ci = frozenset({lit("P12"), lit("P21")})
        cj = frozenset({lit("P21", False)})
        self.assertEqual(resolver(ci, cj), [frozenset({lit("P12")})])

    def test_resolvente_tautologico_descartado(self):
        ci = frozenset({lit("A"), lit("B")})
        cj = frozenset({lit("A", False), lit("B", False)})
        self.assertEqual(resolver(ci, cj), [])

    def test_modus_ponens(self):
        kb = para_cnf(Implica("A", "B")) | para_cnf("A")
        self.assertTrue(pl_resolucao(kb, "B"))
        self.assertFalse(pl_resolucao(kb, Nao("B")))


class TestExemploDoLivro(unittest.TestCase):
    """Russell & Norvig, seção 7.4: em (1,1) sem brisa ⇒ (1,2) e (2,1) sem poço."""

    def montar(self):
        bc = BaseConhecimento()
        for a in axiomas_iniciais():
            bc.tell(a)
        for casa in [(1, 1), (2, 1), (1, 2)]:
            for a in axiomas_da_casa(casa):
                bc.tell(a)
        return bc

    def test_sem_brisa_em_1_1(self):
        bc = self.montar()
        bc.tell(Nao(B(1, 1)))
        self.assertTrue(bc.ask(Nao(P(1, 2))))
        self.assertTrue(bc.ask(Nao(P(2, 1))))

    def test_brisa_em_2_1_nao_prova_nada_sozinha(self):
        bc = self.montar()
        bc.tell(Nao(B(1, 1)))
        bc.tell(B(2, 1))
        self.assertFalse(bc.ask(P(3, 1)))
        self.assertFalse(bc.ask(Nao(P(3, 1))))

    def test_figura_7_4_localiza_wumpus_e_poco(self):
        # (1,1): nada; (2,1): brisa; (1,2): fedor, sem brisa
        bc = self.montar()
        for s in [Nao(B(1, 1)), Nao(F(1, 1)), B(2, 1), Nao(F(2, 1)),
                  Nao(B(1, 2)), F(1, 2)]:
            bc.tell(s)
        self.assertTrue(bc.ask(W(1, 3)))        # o Wumpus está em (1,3)
        self.assertTrue(bc.ask(P(3, 1)))        # o poço está em (3,1)
        self.assertTrue(bc.ask(sentenca_segura((2, 2))))

    def test_grito_nao_torna_bc_inconsistente(self):
        """Regressão do bug da versão anterior: depois do grito a BC não pode provar tudo."""
        bc = self.montar()
        bc.tell(Nao(B(1, 1)))
        bc.tell(B(2, 1))
        bc.tell(M)
        self.assertFalse(bc.ask(Nao(P(3, 1))))  # continua sem saber do poço
        self.assertTrue(bc.ask(Ou(Nao(W(3, 1)), M)))


@unittest.skipIf(sympy is None, "sympy não instalado")
class TestContraSympy(unittest.TestCase):
    """Compara nossa CNF e nossa resolução com o sympy em fórmulas aleatórias."""

    SIMBOLOS = ["A", "B", "C", "D"]

    def formula(self, rng, profundidade):
        if profundidade == 0 or rng.random() < 0.3:
            return rng.choice(self.SIMBOLOS)
        op = rng.choice(["¬", "∧", "∨", "⇒", "⇔"])
        if op == "¬":
            return Nao(self.formula(rng, profundidade - 1))
        return (op, self.formula(rng, profundidade - 1), self.formula(rng, profundidade - 1))

    def para_sympy(self, s):
        if isinstance(s, str):
            return sympy.Symbol(s)
        op, *a = s
        a = [self.para_sympy(x) for x in a]
        return {"¬": lambda: sympy.Not(a[0]), "∧": lambda: sympy.And(*a),
                "∨": lambda: sympy.Or(*a), "⇒": lambda: sympy.Implies(*a),
                "⇔": lambda: sympy.Equivalent(*a)}[op]()

    def clausulas_para_sympy(self, clausulas):
        return sympy.And(*[sympy.Or(*[sympy.Symbol(s) if pos else sympy.Not(sympy.Symbol(s))
                                      for s, pos in c]) for c in clausulas])

    def test_cnf_equivalente(self):
        rng = random.Random(0)
        for _ in range(200):
            f = self.formula(rng, 4)
            nossa = self.clausulas_para_sympy(para_cnf(f))
            original = self.para_sympy(f)
            self.assertFalse(satisfiable(sympy.Xor(nossa, original)), f)

    def test_consequencia_logica(self):
        rng = random.Random(1)
        for _ in range(1000):
            kb = [self.formula(rng, 3) for _ in range(3)]
            alfa = self.formula(rng, 2)
            kb_sympy = sympy.And(*[self.para_sympy(k) for k in kb])
            # A estratégia do conjunto de suporte só é completa quando a BC é
            # satisfatível. A BC do Wumpus sempre é (descreve um mundo real);
            # BCs contraditórias geradas ao acaso ficam fora deste teste.
            if not satisfiable(kb_sympy):
                continue
            clausulas = set().union(*[para_cnf(k) for k in kb])
            nossa = pl_resolucao(clausulas, alfa)
            verdade = not satisfiable(sympy.And(kb_sympy, sympy.Not(self.para_sympy(alfa))))
            self.assertEqual(nossa, verdade, (kb, alfa))


class TestAgentes(unittest.TestCase):
    def test_b_e_c_resolvem_o_mundo_classico(self):
        for cls in (AgenteLogico, AgenteObjetivo):
            r = rodar_episodio(cls, mundo=MundoWumpus.classico())
            self.assertEqual(r.desfecho, "saiu_com_ouro", cls.nome)

    def test_b_nunca_morre(self):
        """B só entra em casa provada segura: não pode morrer em nenhum mundo."""
        for s in range(300):
            self.assertFalse(rodar_episodio(AgenteLogico, s).morte, s)

    def test_b_sem_posicao(self):
        for s in range(100):
            r = rodar_episodio(AgenteLogico, s, fornecer_posicao=False)
            self.assertFalse(r.morte or r.erro, s)

    def test_a_nao_guarda_estado(self):
        a = AgenteReativo()
        antes = dict(vars(a))
        for _ in range(50):
            a.agir(Percepcao(brisa=True, posicao=(2, 1)))
        self.assertEqual(vars(a), antes)
        self.assertIs(a.agir(Percepcao(brilho=True, posicao=(3, 3))), Acao.AGARRAR)


if __name__ == "__main__":
    unittest.main()


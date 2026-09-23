"""AGENTE C — Lógico + objetivo, com planejamento por BFS (Aulas 3, 4 e 6).  [0,6 ponto]

Reaproveita o Agente B (mesma BC, mesmo TELL/ASK por resolução) para saber ONDE
é seguro e usa busca em largura (BFS) para planejar COMO chegar lá.

Máquina de estados (slide 10):
  EXPLORAR  → BFS até a casa segura não visitada mais próxima
  AGARRAR   → brilho na casa atual
  VOLTAR    → BFS até (1,1) passando apenas por casas provadas seguras
  SAIR      → chegou em (1,1) com o ouro (ou decidiu desistir)

Quando não há casa segura nova e o ouro não apareceu (o dilema do risco), C
tenta, nesta ordem:
  1. FLECHA: se existe uma casa de fronteira que a BC prova sem poço (¬P) mas
     não consegue provar sem Wumpus, C vai até uma casa vizinha já visitada e
     atira nela. Com grito, M entra na BC; sem grito, ¬W entra para todo o raio.
     Nos dois casos a casa-alvo passa a ser PROVADA segura: o tiro sempre compra
     uma casa nova por 10 pontos + deslocamento.
  2. RISCO: estima P(morte) de cada casa de fronteira por enumeração dos modelos
     consistentes com a BC (inferência probabilística, R&N cap. 12/13) e entra na
     menos perigosa só se a utilidade esperada for maior que a de sair.
  3. SAIR: volta a (1,1) por BFS e sai.
"""

from __future__ import annotations

from collections import deque

from wumpus import Acao, Percepcao, vizinhos

from agentes.agente_logico import (INICIO, M, AgenteLogico, P, W,
                                   todas_as_casas)

PROB_POCO = 0.2          # prior do enunciado
# G = valor de continuar explorando, se sobreviver ao passo arriscado.
# A estimativa a priori (G ≈ 500, limiar 0,33) foi testada nas sementes
# 1000..2999 — NÃO nas públicas 0..199, que ficam como conjunto de relato — e
# perdeu ~20 pontos por partida para limiares entre 0,20 e 0,30: as casas com
# p ≈ 1/3 matavam mais do que rendiam. G = 350 (limiar ≈ 0,26) fica no meio da
# faixa boa, sem ajuste fino. Ver seção de análise do relatório.
GANHO_ESPERADO = 350
PENALIDADE_MORTE = 1000


class AgenteObjetivo(AgenteLogico):
    nome = "C · Lógico + BFS"

    def __init__(self):
        super().__init__()
        self.fase = "explorar"
        self.plano = []            # fila de casas a percorrer
        self.mira = None           # casa-alvo da flecha, quando há plano de tiro
        self.nos_expandidos = 0
        self.riscos_assumidos = 0
        # Arrisca se p < G / (G + 1000): EU(arriscar) = (1-p)·G − p·1000 > 0 = EU(sair)
        self.limiar_risco = GANHO_ESPERADO / (GANHO_ESPERADO + PENALIDADE_MORTE)

    # ------------------------------------------------------------ BFS
    def bfs(self, origem, destinos, permitidas):
        """Menor caminho (lista de casas) de `origem` até alguma casa de `destinos`,
        andando só por casas em `permitidas`. Devolve None se não houver caminho.

        Todas as arestas custam 1 AVANCAR, então a BFS é ótima em número de casas.
        """
        destinos = set(destinos)
        if origem in destinos:
            return [origem]
        fila = deque([origem])
        pai = {origem: None}
        while fila:
            atual = fila.popleft()
            self.nos_expandidos += 1
            for viz in vizinhos(atual, self.tamanho):
                if viz in pai or (viz not in permitidas and viz not in destinos):
                    continue
                pai[viz] = atual
                if viz in destinos:
                    caminho = [viz]
                    while pai[caminho[-1]] is not None:
                        caminho.append(pai[caminho[-1]])
                    return caminho[::-1]
                fila.append(viz)
        return None

    def _ir_para(self, destinos, permitidas=None) -> bool:
        permitidas = self.visitadas if permitidas is None else permitidas
        caminho = self.bfs(self.pos, destinos, permitidas)
        if caminho is None:
            return False
        self.caminho = caminho[1:]
        return True

    # ------------------------------------------------------------ fronteira
    def _fronteira(self):
        """Casas não visitadas vizinhas de alguma casa visitada."""
        return {v for c in self.visitadas for v in vizinhos(c, self.tamanho)
                if v not in self.visitadas}

    # ------------------------------------------------------------ decisão
    def _decidir(self, p: Percepcao) -> Acao:
        if p.brilho and not self.tem_ouro:
            self.caminho, self.mira = [], None
            self.fase = "voltar"
            return Acao.AGARRAR

        # plano de tiro: já na casa de onde se atira?
        while self.caminho and self.caminho[0] == self.pos:
            self.caminho.pop(0)
        if self.mira is not None and not self.caminho:
            return self._mirar_e_atirar()

        passo = self._seguir_caminho()
        if passo is not None:
            return passo

        if self.tem_ouro:
            if self.pos == INICIO:
                return Acao.SAIR
            self._ir_para({INICIO})
            return self._seguir_caminho()

        # EXPLORAR: fronteira provada segura, a mais próxima por BFS
        fronteira = self._fronteira()
        seguras = {c for c in fronteira if self.e_segura(c)}
        if seguras and self._ir_para(seguras, self.visitadas):
            self.fase = "explorar"
            return self._seguir_caminho()

        # FLECHA
        if self.tem_flecha:
            alvo = self._alvo_da_flecha(fronteira)
            if alvo is not None:
                casa_tiro = self._casa_de_tiro(alvo)
                if casa_tiro is not None and self._ir_para({casa_tiro}):
                    self.fase, self.mira = "atirar", alvo
                    passo = self._seguir_caminho()
                    return passo if passo is not None else self._mirar_e_atirar()

        # RISCO
        if fronteira:
            riscos = self.probabilidade_de_morte(fronteira)
            melhor = min(fronteira, key=lambda c: (riscos[c], c))
            if riscos[melhor] < self.limiar_risco:
                origem = self.bfs(self.pos, {v for v in vizinhos(melhor, self.tamanho)
                                             if v in self.visitadas}, self.visitadas)
                if origem is not None:
                    self.caminho = origem[1:] + [melhor]
                    self.fase = "arriscar"
                    self.riscos_assumidos += 1
                    return self._seguir_caminho()

        # SAIR
        self.fase = "voltar"
        if self.pos == INICIO:
            return Acao.SAIR
        self._ir_para({INICIO})
        return self._seguir_caminho()

    # ------------------------------------------------------------ flecha
    def _alvo_da_flecha(self, fronteira):
        """Casa de fronteira em que o ÚNICO perigo possível é o Wumpus (BC ⊨ ¬P, BC ⊭ ¬W)."""
        candidatas = [c for c in fronteira
                      if self.bc.ask(("¬", P(*c))) and not self.bc.ask(("∨", ("¬", W(*c)), M))]
        if not candidatas:
            return None
        # prefere a casa em que a BC PROVA o Wumpus; depois, a mais próxima
        return min(candidatas, key=lambda c: (not self.bc.ask(W(*c)),
                                              abs(c[0] - self.pos[0]) + abs(c[1] - self.pos[1])))

    def _casa_de_tiro(self, alvo):
        origens = [v for v in vizinhos(alvo, self.tamanho) if v in self.visitadas]
        if not origens:
            return None
        caminho = self.bfs(self.pos, set(origens), self.visitadas)
        return caminho[-1] if caminho else None

    def _mirar_e_atirar(self) -> Acao:
        alvo = self.mira
        delta = (alvo[0] - self.pos[0], alvo[1] - self.pos[1])
        desejada = type(self.direcao).de_delta(delta)
        if self.direcao == desejada:
            self.mira = None
            return Acao.ATIRAR
        return Acao.VIRAR_DIREITA if self.direcao.direita() == desejada else Acao.VIRAR_ESQUERDA

    # ------------------------------------------------------------ risco
    def probabilidade_de_morte(self, casas) -> dict:
        """P(morrer ao entrar em cada casa), dado TUDO o que está na BC.

        Enumeração de modelos restrita ao que importa: os símbolos de poço que
        aparecem na BC e ainda não são fatos conhecidos (a fronteira) e a posição
        do Wumpus. Poços e Wumpus são independentes a priori (poço: 0,2 por casa;
        Wumpus: uniforme fora de (1,1)) e as cláusulas de P e de W não se misturam,
        então as duas enumerações são feitas separadamente e combinadas no fim.
        Cada modelo candidato é conferido contra as cláusulas da BC.
        """
        fatos = self.bc.fatos()
        clausulas = self.bc.clausulas

        # --- poços ---
        clausulas_p = [c for c in clausulas if any(s[0] in "PB" for s, _ in c)]
        livres = sorted({s for c in clausulas_p for s, _ in c
                         if s.startswith("P") and s not in fatos})
        peso_total = 0.0
        peso_poco = {s: 0.0 for s in livres}
        for mascara in range(1 << len(livres)):
            modelo = dict(fatos)
            k = 0
            for i, s in enumerate(livres):
                v = bool(mascara >> i & 1)
                modelo[s] = v
                k += v
            if all(any(modelo.get(s, False) == pos for s, pos in c) for c in clausulas_p):
                peso = PROB_POCO ** k * (1 - PROB_POCO) ** (len(livres) - k)
                peso_total += peso
                for s in livres:
                    if modelo[s]:
                        peso_poco[s] += peso

        def p_poco(casa):
            s = P(*casa)
            if s in fatos:
                return 1.0 if fatos[s] else 0.0
            if s in peso_poco and peso_total > 0:
                return peso_poco[s] / peso_total
            return PROB_POCO  # casa sobre a qual a BC não diz nada

        # --- Wumpus ---
        wumpus_morto = fatos.get(M, False)
        clausulas_w = [c for c in clausulas if any(s[0] in "WF" for s, _ in c)]
        casas_w = [c for c in todas_as_casas(self.tamanho) if c != INICIO]
        consistentes = []
        for local in casas_w:
            modelo = dict(fatos)
            for c in todas_as_casas(self.tamanho):
                modelo[W(*c)] = (c == local)
            if all(any(modelo.get(s, False) == pos for s, pos in cl) for cl in clausulas_w):
                consistentes.append(local)

        def p_wumpus(casa):
            if wumpus_morto or not consistentes:
                return 0.0
            return sum(1 for c in consistentes if c == casa) / len(consistentes)

        return {c: 1 - (1 - p_poco(c)) * (1 - p_wumpus(c)) for c in casas}

    def depurar(self) -> str:
        return (f"Fase: {self.fase} | caminho: {self.caminho} | "
                f"cláusulas: {len(self.bc)} | nós BFS: {self.nos_expandidos}")

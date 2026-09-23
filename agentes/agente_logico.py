"""AGENTE B — Agente lógico baseado em conhecimento (Aula 6).        [1,3 ponto]

Ciclo: PERCEPÇÃO → TELL → BC → ASK (resolução) → AÇÃO

Representação (decisão do grupo)
--------------------------------
Sentença : átomo (str, ex. "P(1,3)") ou tupla (operador, *argumentos), com
           operadores "¬", "∧", "∨", "⇒", "⇔". Há construtores Nao/E/Ou/Implica/Equivale.
Literal  : (simbolo, positivo)  — ("P(1,3)", False) é ¬P(1,3).
Cláusula : frozenset de literais (disjunção).  BC: conjunto de cláusulas (conjunção).

Símbolos do mundo
-----------------
P(x,y)  há poço em (x,y)          B(x,y)  há brisa em (x,y)
W(x,y)  o Wumpus está em (x,y)    F(x,y)  há fedor em (x,y)
M       o Wumpus está morto (ouvimos o grito)

W(x,y) é a LOCALIZAÇÃO do Wumpus, vivo ou morto. Isso importa: no simulador
oficial o fedor continua depois que o Wumpus morre. Uma versão anterior deste
agente fazia tell(¬W) em todas as casas ao ouvir o grito; como a BC também diz
"existe exatamente um Wumpus", a BC ficava contraditória e passava a "provar"
que qualquer casa era segura (de uma BC inconsistente se deduz qualquer coisa).
A correção é modelar a morte com um símbolo próprio, M, e definir

    Segura(x,y)  ≡  ¬P(x,y) ∧ (¬W(x,y) ∨ M)

Axiomas (gerados por código, idênticos para toda casa)
------------------------------------------------------
    ¬P(1,1)   ¬W(1,1)
    W(1,1) ∨ W(1,2) ∨ ... ∨ W(4,4)                 (pelo menos um Wumpus)
    ¬(W(a) ∧ W(b))  para todo par a ≠ b             (no máximo um Wumpus)
    B(x,y) ⇔ ∨ P(vizinhos)
    F(x,y) ⇔ W(x,y) ∨ ∨ W(vizinhos)                 (fedor inclui a própria casa)

Seguindo a dica do slide 09, os bicondicionais de B e F só são instanciados
para as casas visitadas, e as respostas de ask() ficam em cache até a BC mudar.
"""

from __future__ import annotations

import heapq
from itertools import combinations, product

from wumpus import Acao, Agente, Direcao, Percepcao, acoes_para_vizinho, vizinhos

TAMANHO = 4
INICIO = (1, 1)


# ==========================================================================
# Sentenças
# ==========================================================================

def Nao(s):
    return ("¬", s)


def E(*args):
    return args[0] if len(args) == 1 else ("∧",) + tuple(args)


def Ou(*args):
    return args[0] if len(args) == 1 else ("∨",) + tuple(args)


def Implica(a, b):
    return ("⇒", a, b)


def Equivale(a, b):
    return ("⇔", a, b)


def P(x, y): return f"P({x},{y})"
def W(x, y): return f"W({x},{y})"
def B(x, y): return f"B({x},{y})"
def F(x, y): return f"F({x},{y})"


M = "M"  # Wumpus morto


def texto(s) -> str:
    """Sentença em notação legível (para depuração e relatório)."""
    if isinstance(s, str):
        return s
    op, *args = s
    if op == "¬":
        return "¬" + texto(args[0])
    return "(" + f" {op} ".join(texto(a) for a in args) + ")"


def texto_clausula(c) -> str:
    if not c:
        return "□"
    return " ∨ ".join(("" if pos else "¬") + sim for sim, pos in sorted(c))


# ==========================================================================
# Conversão para CNF
# ==========================================================================

def _eliminar_implicacoes(s):
    """Passo 1: α ⇔ β  vira (¬α ∨ β) ∧ (¬β ∨ α);   α ⇒ β  vira ¬α ∨ β."""
    if isinstance(s, str):
        return s
    op, *args = s
    args = [_eliminar_implicacoes(a) for a in args]
    if op == "⇒":
        a, b = args
        return ("∨", ("¬", a), b)
    if op == "⇔":
        a, b = args
        return ("∧", ("∨", ("¬", a), b), ("∨", ("¬", b), a))
    return (op, *args)


def _mover_negacao(s, negar=False):
    """Passo 2: empurra ¬ até os átomos (De Morgan e dupla negação)."""
    if isinstance(s, str):
        return ("¬", s) if negar else s
    op, *args = s
    if op == "¬":
        return _mover_negacao(args[0], not negar)
    if op in ("∧", "∨"):
        novo_op = op if not negar else ("∨" if op == "∧" else "∧")
        return (novo_op, *[_mover_negacao(a, negar) for a in args])
    raise ValueError(f"Operador inesperado após eliminar ⇒/⇔: {op}")


def e_tautologia(c) -> bool:
    return any((sim, not pos) in c for sim, pos in c)


def _distribuir(s) -> list:
    """Passo 3: distribui ∨ sobre ∧ e devolve a lista de cláusulas."""
    if isinstance(s, str):
        return [frozenset({(s, True)})]
    op, *args = s
    if op == "¬":  # após o passo 2, só aparece aplicado a átomo
        return [frozenset({(args[0], False)})]
    partes = [_distribuir(a) for a in args]
    if op == "∧":
        return [c for p in partes for c in p]
    # op == "∨": produto cartesiano das cláusulas de cada disjunto
    saida = []
    for combinacao in product(*partes):
        c = frozenset().union(*combinacao)
        if not e_tautologia(c):
            saida.append(c)
    return saida


def _remover_subsumidas(clausulas) -> set:
    ordenadas = sorted(set(clausulas), key=len)
    mantidas = []
    for c in ordenadas:
        if not any(k <= c for k in mantidas):
            mantidas.append(c)
    return set(mantidas)


def para_cnf(sentenca) -> set:
    """Eliminar ⇔ e ⇒, empurrar ¬ para dentro (De Morgan), distribuir ∨ sobre ∧."""
    s = _eliminar_implicacoes(sentenca)
    s = _mover_negacao(s)
    return _remover_subsumidas(_distribuir(s))


# ==========================================================================
# Resolução
# ==========================================================================

def resolver(ci, cj) -> list:
    """Todos os resolventes possíveis entre as cláusulas ci e cj (tautologias descartadas)."""
    saida = []
    for sim, pos in ci:
        if (sim, not pos) in cj:
            r = (ci - {(sim, pos)}) | (cj - {(sim, not pos)})
            if not e_tautologia(r):
                saida.append(frozenset(r))
    return saida


def _simbolos(c):
    return {sim for sim, _ in c}


def _relevantes(clausulas, simbolos_alvo) -> list:
    """Cláusulas ligadas (direta ou indiretamente) aos símbolos da consulta.

    Uma cláusula sem nenhum símbolo em comum com esse componente conexo nunca
    participa de uma refutação dele. Cortar o resto não muda a resposta, só o custo.
    """
    alvo = set(simbolos_alvo)
    restantes = list(clausulas)
    selecionadas = []
    mudou = True
    while mudou:
        mudou = False
        ainda = []
        for c in restantes:
            if _simbolos(c) & alvo:
                selecionadas.append(c)
                alvo |= _simbolos(c)
                mudou = True
            else:
                ainda.append(c)
        restantes = ainda
    return selecionadas


class Estatisticas:
    def __init__(self):
        self.resolventes = 0
        self.orcamento_estourado = 0


def pl_resolucao(clausulas, consulta, max_resolventes: int = 20000,
                 estat: Estatisticas | None = None) -> bool:
    """PL-RESOLUTION (Russell & Norvig, fig. 7.12), por refutação.

    Prova BC ⊨ α mostrando que BC ∧ ¬α deriva a cláusula vazia.

    Duas escolhas de implementação, ambas preservando a correção:
      * Conjunto de suporte: toda resolução envolve ao menos uma cláusula que
        descende de ¬α. Se a BC é satisfatível (é: descreve o mundo real), a
        estratégia continua completa por refutação.
      * Preferência por cláusulas curtas (a menor cláusula do suporte é
        processada primeiro); cláusulas unitárias fecham a prova depressa.

    O teto `max_resolventes` é uma trava de segurança. Se for atingido, a função
    responde False ("não provei") — o agente trata a casa como NÃO segura. Pode
    perder uma exploração, nunca causa uma morte.
    """
    negada = para_cnf(Nao(consulta))
    if not negada:
        # CNF vazia = ¬α é tautologia (α é contraditória): BC ∧ ¬α ≡ BC, que é
        # satisfatível, logo BC ⊭ α.
        return False
    alvo = set().union(*(_simbolos(c) for c in negada))
    usaveis = _relevantes(clausulas, alvo)

    vistas = set(usaveis)
    fila = []  # heap (tamanho, contador, cláusula)
    contador = 0
    for c in negada:
        if c not in vistas:
            vistas.add(c)
            heapq.heappush(fila, (len(c), contador, c))
            contador += 1
    processadas = []

    while fila:
        _, _, dada = heapq.heappop(fila)
        if any(k <= dada and k != dada for k in processadas):
            continue  # subsumida por uma cláusula já processada
        for outra in usaveis + processadas:
            for r in resolver(dada, outra):
                if not r:
                    return True  # cláusula vazia: BC ∧ ¬α é insatisfatível
                if r in vistas:
                    continue
                if any(k <= r for k in usaveis if len(k) < len(r)):
                    continue  # subsumida pela BC: não acrescenta nada
                vistas.add(r)
                heapq.heappush(fila, (len(r), contador, r))
                contador += 1
                if estat is not None:
                    estat.resolventes += 1
                if contador > max_resolventes:
                    if estat is not None:
                        estat.orcamento_estourado += 1
                    return False
        processadas.append(dada)
    return False  # nenhuma contradição: BC ⊭ α


# ==========================================================================
# Base de conhecimento
# ==========================================================================

class BaseConhecimento:
    """BC proposicional em CNF com TELL/ASK."""

    def __init__(self):
        self.clausulas = set()
        self._cache = {}
        self.estat = Estatisticas()
        self.consultas = 0
        self.consultas_resolvidas = 0  # as que chegaram a rodar a resolução

    def tell(self, sentenca) -> None:
        """Converte a sentença para CNF e acrescenta as cláusulas à BC."""
        novas = para_cnf(sentenca) - self.clausulas
        if novas:
            self.clausulas |= novas
            self._cache.clear()  # a BC mudou: respostas antigas podem ter mudado

    def ask(self, consulta) -> bool:
        """True se BC ⊨ consulta, provado por resolução por refutação.

        Uma conjunção é quebrada em partes: BC ⊨ α ∧ β  sse  BC ⊨ α e BC ⊨ β.
        (Mantém cada refutação pequena; o resultado é o mesmo.)
        """
        self.consultas += 1
        if not isinstance(consulta, str) and consulta[0] == "∧":
            return all(self.ask(parte) for parte in consulta[1:])
        chave = consulta
        if chave not in self._cache:
            self.consultas_resolvidas += 1
            self._cache[chave] = pl_resolucao(self.clausulas, consulta, estat=self.estat)
        return self._cache[chave]

    def fatos(self) -> dict:
        """Literais unitários da BC (símbolo -> valor). Usado só na estimativa de risco do C."""
        return {sim: pos for c in self.clausulas if len(c) == 1 for sim, pos in c}

    def __len__(self):
        return len(self.clausulas)


# ==========================================================================
# Conhecimento específico do Mundo Wumpus
# ==========================================================================

def todas_as_casas(tamanho=TAMANHO):
    return [(x, y) for x in range(1, tamanho + 1) for y in range(1, tamanho + 1)]


def axiomas_iniciais(tamanho=TAMANHO) -> list:
    casas = todas_as_casas(tamanho)
    axiomas = [Nao(P(*INICIO)), Nao(W(*INICIO)),
               Ou(*[W(*c) for c in casas])]                       # pelo menos um Wumpus
    for a, b in combinations(casas, 2):                            # no máximo um Wumpus
        axiomas.append(Nao(E(W(*a), W(*b))))
    return axiomas


def axiomas_da_casa(casa, tamanho=TAMANHO) -> list:
    """Regras de brisa e fedor de UMA casa, geradas pela mesma função para qualquer casa."""
    viz = vizinhos(casa, tamanho)
    return [
        Equivale(B(*casa), Ou(*[P(*v) for v in viz])),
        Equivale(F(*casa), Ou(W(*casa), *[W(*v) for v in viz])),
    ]


def sentenca_segura(casa):
    return E(Nao(P(*casa)), Ou(Nao(W(*casa)), M))


def raio_da_flecha(pos, direcao, tamanho=TAMANHO) -> list:
    dx, dy = direcao.delta
    x, y = pos[0] + dx, pos[1] + dy
    casas = []
    while 1 <= x <= tamanho and 1 <= y <= tamanho:
        casas.append((x, y))
        x, y = x + dx, y + dy
    return casas


# ==========================================================================
# O agente
# ==========================================================================

class AgenteLogico(Agente):
    """Agente B: explora só casas PROVADAS seguras, com retrocesso pelo próprio rastro.

    B é local: a cada casa pergunta à BC sobre os vizinhos imediatos. Não planeja
    rotas; quando não há vizinho novo e seguro, volta pelo caminho que fez
    (busca em profundidade online, R&N seção 4.5). Não usa a flecha e nunca
    arrisca: se nada for provadamente seguro, volta a (1,1) e sai.
    """

    nome = "B · Lógico"

    def __init__(self):
        super().__init__()
        self.tamanho = TAMANHO
        self.bc = BaseConhecimento()
        for axioma in axiomas_iniciais(self.tamanho):
            self.bc.tell(axioma)

        # estado interno (memória da partida)
        self.pos = INICIO
        self.direcao = Direcao.LESTE
        self.ultima_acao = None
        self.visitadas = {INICIO}
        self.com_axiomas = set()
        self.seguras = {INICIO}
        self.tem_ouro = False
        self.tem_flecha = True
        self.raio_pendente = None  # casas que a última flecha atravessou
        self.rastro = []           # pilha de casas para o retrocesso
        self.caminho = []          # próximas casas a percorrer

    # ------------------------------------------------------------ estado
    def _atualizar_posicao(self, p: Percepcao) -> None:
        if p.posicao is not None:
            self.pos, self.direcao = tuple(p.posicao), p.direcao
            return
        # Sem propriocepção: rastreia pela última ação (e pelo baque).
        a = self.ultima_acao
        if a is Acao.VIRAR_ESQUERDA:
            self.direcao = self.direcao.esquerda()
        elif a is Acao.VIRAR_DIREITA:
            self.direcao = self.direcao.direita()
        elif a is Acao.AVANCAR and not p.baque:
            dx, dy = self.direcao.delta
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def _tell_percepcao(self, p: Percepcao) -> None:
        """Tudo o que o agente aprende entra na BC por aqui, e só por tell()."""
        casa = self.pos
        self.visitadas.add(casa)
        if casa not in self.com_axiomas:
            self.com_axiomas.add(casa)
            for axioma in axiomas_da_casa(casa, self.tamanho):
                self.bc.tell(axioma)
        self.bc.tell(B(*casa) if p.brisa else Nao(B(*casa)))
        self.bc.tell(F(*casa) if p.fedor else Nao(F(*casa)))
        # estou vivo nesta casa: não há poço e, se o Wumpus está aqui, está morto
        self.bc.tell(sentenca_segura(casa))
        if self.raio_pendente is not None:
            if p.grito:
                self.bc.tell(M)
            else:
                # a flecha cruzou o raio sem grito: o Wumpus não está em nenhuma dessas casas
                self.bc.tell(E(*[Nao(W(*c)) for c in self.raio_pendente]))
            self.raio_pendente = None

    def e_segura(self, casa) -> bool:
        """Consulta a BC (resolução). Segurança é monotônica: uma vez provada, fica."""
        if casa in self.seguras:
            return True
        if self.bc.ask(sentenca_segura(casa)):
            self.seguras.add(casa)
            return True
        return False

    def _registrar(self, acao: Acao) -> Acao:
        self.ultima_acao = acao
        if acao is Acao.AGARRAR:
            self.tem_ouro = True
        elif acao is Acao.ATIRAR and self.tem_flecha:
            self.tem_flecha = False
            self.raio_pendente = raio_da_flecha(self.pos, self.direcao, self.tamanho)
        return acao

    def _passo_para(self, destino) -> Acao:
        """Primeira ação (giro ou AVANCAR) para ir a uma casa adjacente."""
        return acoes_para_vizinho(self.pos, self.direcao, destino)[0]

    def _seguir_caminho(self):
        while self.caminho and self.caminho[0] == self.pos:
            self.caminho.pop(0)
        if self.caminho:
            return self._passo_para(self.caminho[0])
        return None

    def _custo_giro(self, destino) -> int:
        return len(acoes_para_vizinho(self.pos, self.direcao, destino))

    # ------------------------------------------------------------ ciclo
    def agir(self, percepcao: Percepcao) -> Acao:
        self._atualizar_posicao(percepcao)
        self._tell_percepcao(percepcao)
        return self._registrar(self._decidir(percepcao))

    def _decidir(self, p: Percepcao) -> Acao:
        if p.brilho and not self.tem_ouro:
            return Acao.AGARRAR

        passo = self._seguir_caminho()
        if passo is not None:
            return passo

        if self.tem_ouro:
            if self.pos == INICIO:
                return Acao.SAIR
            return self._recuar()

        novas = [v for v in vizinhos(self.pos, self.tamanho)
                 if v not in self.visitadas and self.e_segura(v)]
        if novas:
            alvo = min(novas, key=self._custo_giro)
            self.rastro.append(self.pos)
            self.caminho = [alvo]
            return self._seguir_caminho()

        if self.pos == INICIO and not self.rastro:
            return Acao.SAIR  # explorou tudo o que era provadamente seguro
        return self._recuar()

    def _recuar(self) -> Acao:
        anterior = self.rastro.pop()
        self.caminho = [anterior]
        return self._seguir_caminho()

    def depurar(self) -> str:
        seguras = sorted(c for c in self.seguras if c not in self.visitadas)
        return (f"Cláusulas na BC: {len(self.bc)} | seguras não visitadas: {seguras} | "
                f"ouro: {'sim' if self.tem_ouro else 'não'}")

"""AGENTE D (bônus) — LLM como raciocinador.                          [+0,5 bônus]

Pergunta do slide 14: "Um LLM consegue manter um raciocínio consistente com
informação incompleta?"

Como funciona
-------------
PERCEPÇÕES em texto → LLM raciocina → AÇÃO devolvida ao jogo.

O agente guarda só OBSERVAÇÕES (o que sentiu em cada casa visitada, se tem o
ouro, a flecha, o grito). Nenhuma inferência é feita pelo código: a cada decisão
o LLM recebe esse resumo em texto e devolve um JSON com
  - "casas_seguras" / "casas_perigosas": o que ele AFIRMA saber;
  - "acao": MOVER (para uma casa vizinha), AGARRAR, ATIRAR (numa direção) ou SAIR.
O código só traduz MOVER/ATIRAR em giros + AVANCAR/ATIRAR.

Medição (para comparar com o Agente B nas mesmas sementes)
-----------------------------------------------------------
Uma BC "sombra", idêntica à do Agente B, recebe as mesmas percepções. Ela NÃO
decide nada; serve de juiz:
  - movimento inseguro: o LLM entrou numa casa que a lógica não prova segura;
  - alucinação: o LLM afirmou algo que a BC prova ser falso (chamou de segura
    uma casa com poço/Wumpus provado, ou de perigosa uma casa provada segura);
  - afirmação sem prova: chamou de segura uma casa que a BC não consegue provar;
  - custo: tokens de entrada/saída e tempo de cada partida.

Configuração (variáveis de ambiente ou arquivo .env na raiz do projeto)
------------------------------------------------------------------------
  Uma destas chaves (a primeira encontrada define o provedor):
    XAI_API_KEY          Grok (xAI)            modelo padrão grok-4.7
    DASHSCOPE_API_KEY    Qwen (Alibaba Cloud)  modelo padrão qwen-plus
    ANTHROPIC_API_KEY    Claude (Anthropic)    modelo padrão claude-haiku-4-5-20251001
  WUMPUS_LLM_PROVEDOR  força o provedor: xai | alibaba | anthropic
  WUMPUS_LLM_MODELO    troca o modelo
  WUMPUS_LLM_URL       troca o endereço (ex.: DashScope da China,
                       https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions)
  WUMPUS_LLM_LOG       se definido, grava cada chamada (prompt/resposta) neste .jsonl

Só usa a biblioteca padrão (urllib), como o resto do projeto.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from wumpus import Acao, Agente, Direcao, Percepcao, acoes_para_vizinho, vizinhos

from agentes.agente_logico import (INICIO, M, TAMANHO, B, BaseConhecimento, F, Nao,
                                   P, W, E, axiomas_da_casa, axiomas_iniciais,
                                   raio_da_flecha, sentenca_segura)

MAX_CHAMADAS = 60  # trava de custo por partida

SISTEMA = """Você é um agente jogando o Mundo Wumpus (Russell & Norvig).

Regras:
- Grade 4x4, coordenadas (x,y) de 1 a 4. (1,1) é o canto inferior esquerdo; x cresce para a direita (LESTE), y cresce para cima (NORTE).
- Há 1 Wumpus, 1 ouro e cada casa, exceto (1,1), tem poço com probabilidade 0,2. O ouro pode estar numa casa com poço ou com o Wumpus.
- Entrar numa casa com poço ou com o Wumpus vivo mata (−1000). Sair em (1,1) com o ouro vale +1000. Cada ação custa −1; a flecha custa −10 extra.
- Brisa numa casa: existe poço em pelo menos uma casa vizinha (N, S, L, O). Sem brisa: nenhuma vizinha tem poço.
- Fedor numa casa: o Wumpus está nela ou numa vizinha. Sem fedor: não está em nenhuma delas. O fedor continua depois que o Wumpus morre.
- Brilho: o ouro está na casa atual.
- Você tem UMA flecha: ela voa em linha reta a partir da sua casa; se acertar o Wumpus, ouve-se um grito.
- SAIR só funciona em (1,1). Sair sem o ouro vale 0 (fora o custo das ações): é melhor do que morrer.

Responda SOMENTE com um objeto JSON, sem texto fora dele, no formato:
{"raciocinio": "<curto>",
 "casas_seguras": [[x,y], ...],
 "casas_perigosas": [[x,y], ...],
 "acao": "MOVER" | "AGARRAR" | "ATIRAR" | "SAIR",
 "destino": [x,y],
 "direcao": "NORTE" | "LESTE" | "SUL" | "OESTE"}
- "casas_seguras": casas NÃO visitadas que você tem CERTEZA de que não têm poço nem Wumpus vivo.
- "casas_perigosas": casas que você tem CERTEZA de que têm poço ou Wumpus vivo.
- "destino" só para MOVER, e deve ser vizinha da casa atual. "direcao" só para ATIRAR."""


# --------------------------------------------------------------------------
# Cliente da API (substituível nos testes)
# --------------------------------------------------------------------------

def _carregar_env() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for linha in env.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if linha and not linha.startswith("#") and "=" in linha:
                k, v = linha.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


PROVEDORES = {
    # nome: (variável da chave, URL, modelo padrão, formato)
    "xai": ("XAI_API_KEY", "https://api.x.ai/v1/chat/completions", "grok-4.7", "openai"),
    "alibaba": ("DASHSCOPE_API_KEY",
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
                "qwen-plus", "openai"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/messages",
                  "claude-haiku-4-5-20251001", "anthropic"),
}


def provedor_configurado() -> str:
    """WUMPUS_LLM_PROVEDOR, ou o primeiro provedor cuja chave estiver definida."""
    _carregar_env()
    escolhido = os.environ.get("WUMPUS_LLM_PROVEDOR", "").lower()
    if escolhido:
        if escolhido not in PROVEDORES:
            raise RuntimeError(f"WUMPUS_LLM_PROVEDOR deve ser um de {list(PROVEDORES)}")
        return escolhido
    for nome, (var, *_resto) in PROVEDORES.items():
        if os.environ.get(var):
            return nome
    raise RuntimeError("Nenhuma chave encontrada: defina XAI_API_KEY, DASHSCOPE_API_KEY "
                       "ou ANTHROPIC_API_KEY (variável de ambiente ou arquivo .env)")


def chamar_llm(sistema: str, usuario: str, modelo: str) -> tuple:
    """Devolve (texto, tokens_entrada, tokens_saida). Temperatura 0 para reprodutibilidade."""
    nome = provedor_configurado()
    var, url, _, formato = PROVEDORES[nome]
    url = os.environ.get("WUMPUS_LLM_URL", url)
    chave = os.environ[var]
    if formato == "anthropic":
        cabecalhos = {"x-api-key": chave, "anthropic-version": "2023-06-01"}
        corpo = {"model": modelo, "max_tokens": 2000, "temperature": 0, "system": sistema,
                 "messages": [{"role": "user", "content": usuario}]}
    else:  # API no formato OpenAI (xAI e Alibaba DashScope)
        cabecalhos = {"Authorization": f"Bearer {chave}"}
        corpo = {"model": modelo, "max_tokens": 2000, "temperature": 0,
                 "messages": [{"role": "system", "content": sistema},
                              {"role": "user", "content": usuario}]}
    cabecalhos["content-type"] = "application/json"
    dados_envio = json.dumps(corpo).encode("utf-8")

    for tentativa in range(5):
        req = urllib.request.Request(url, data=dados_envio, method="POST", headers=cabecalhos)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                dados = json.loads(r.read())
            if formato == "anthropic":
                texto = "".join(b.get("text", "") for b in dados["content"])
                uso = dados.get("usage", {})
                return texto, uso.get("input_tokens", 0), uso.get("output_tokens", 0)
            texto = dados["choices"][0]["message"].get("content") or ""
            uso = dados.get("usage", {})
            return texto, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and tentativa < 4:
                time.sleep(2 ** tentativa)
                continue
            raise RuntimeError(f"{nome} respondeu {e.code}: {e.read()[:300]!r}") from None
        except urllib.error.URLError:
            if tentativa < 4:
                time.sleep(2 ** tentativa)
                continue
            raise


# --------------------------------------------------------------------------
# O agente
# --------------------------------------------------------------------------

def _casa(v):
    try:
        x, y = int(v[0]), int(v[1])
        return (x, y) if 1 <= x <= TAMANHO and 1 <= y <= TAMANHO else None
    except (TypeError, ValueError, IndexError):
        return None


class AgenteLLM(Agente):
    nome = "D · LLM"
    cliente = staticmethod(chamar_llm)  # os testes trocam por um LLM falso

    def __init__(self):
        super().__init__()
        _carregar_env()
        self.modelo = os.environ.get("WUMPUS_LLM_MODELO") or self._modelo_padrao()
        self.log = os.environ.get("WUMPUS_LLM_LOG")

        # memória: só observações
        self.pos, self.direcao = INICIO, Direcao.LESTE
        self.ultima_acao = None
        self.observacoes = {}      # casa -> texto do que sentiu
        self.tem_ouro = False
        self.tem_flecha = True
        self.ouviu_grito = False
        self.tiro = None           # (casa, direção, raio) do disparo
        self.fila = []             # ações já decididas
        self.anterior = None

        # juiz (não decide nada)
        self.sombra = BaseConhecimento()
        for a in axiomas_iniciais():
            self.sombra.tell(a)
        self.com_axiomas = set()

        # métricas
        self.chamadas = 0
        self.tokens_entrada = 0
        self.tokens_saida = 0
        self.tempo_llm = 0.0
        self.respostas_invalidas = 0
        self.movimentos = 0
        self.movimentos_inseguros = 0
        self.afirmacoes = 0
        self.alucinacoes = 0
        self.sem_prova = 0
        self.exemplos_alucinacao = []

    @staticmethod
    def _modelo_padrao() -> str:
        try:
            return PROVEDORES[provedor_configurado()][2]
        except RuntimeError:
            return "sem-chave"

    # ------------------------------------------------------------ estado
    def _atualizar(self, p: Percepcao) -> None:
        if p.posicao is not None:
            self.pos, self.direcao = tuple(p.posicao), p.direcao
        else:
            a = self.ultima_acao
            if a is Acao.VIRAR_ESQUERDA:
                self.direcao = self.direcao.esquerda()
            elif a is Acao.VIRAR_DIREITA:
                self.direcao = self.direcao.direita()
            elif a is Acao.AVANCAR and not p.baque:
                dx, dy = self.direcao.delta
                self.pos = (self.pos[0] + dx, self.pos[1] + dy)

        sentidos = [n for n, v in (("brisa", p.brisa), ("fedor", p.fedor)) if v]
        self.observacoes[self.pos] = ", ".join(sentidos) if sentidos else "nada"

        # BC sombra recebe exatamente o que o agente percebeu
        c = self.pos
        if c not in self.com_axiomas:
            self.com_axiomas.add(c)
            for a in axiomas_da_casa(c):
                self.sombra.tell(a)
        self.sombra.tell(B(*c) if p.brisa else Nao(B(*c)))
        self.sombra.tell(F(*c) if p.fedor else Nao(F(*c)))
        self.sombra.tell(sentenca_segura(c))
        if self.tiro is not None and self.tiro[3] is None:
            ouviu = p.grito
            self.tiro = self.tiro[:3] + (ouviu,)
            if ouviu:
                self.ouviu_grito = True
                self.sombra.tell(M)
            else:
                self.sombra.tell(E(*[Nao(W(*x)) for x in self.tiro[2]]))

    def _texto_estado(self, p: Percepcao) -> str:
        linhas = [f"Você está em {self.pos}, virado para o {self.direcao.name}."]
        agora = [n for n, v in (("brisa", p.brisa), ("fedor", p.fedor), ("BRILHO", p.brilho)) if v]
        linhas.append("Percepção agora: " + (", ".join(agora) if agora else "nada") + ".")
        linhas.append(f"Ouro: {'com você' if self.tem_ouro else 'ainda não pegou'}. "
                      f"Flecha: {'disponível' if self.tem_flecha else 'já usada'}.")
        if self.tiro is not None:
            casa, d, _, grito = self.tiro
            linhas.append(f"Você atirou de {casa} para o {d.name}: "
                          + ("ouviu um GRITO (o Wumpus está morto)." if grito else "nenhum grito."))
        linhas.append("Casas visitadas e o que sentiu em cada uma:")
        for c in sorted(self.observacoes):
            linhas.append(f"  {c}: {self.observacoes[c]}")
        linhas.append(f"Vizinhas da casa atual: {vizinhos(self.pos)}.")
        return "\n".join(linhas)

    # ------------------------------------------------------------ decisão
    def agir(self, percepcao: Percepcao) -> Acao:
        self._atualizar(percepcao)
        if not self.fila:
            self.fila = self._decidir(percepcao)
        acao = self.fila.pop(0)
        self.ultima_acao = acao
        if acao is Acao.AVANCAR:
            self.anterior = self.pos
        elif acao is Acao.AGARRAR and percepcao.brilho:
            self.tem_ouro = True
        elif acao is Acao.ATIRAR and self.tem_flecha:
            self.tem_flecha = False
            self.tiro = (self.pos, self.direcao, raio_da_flecha(self.pos, self.direcao), None)
        return acao

    def _decidir(self, p: Percepcao) -> list:
        if self.chamadas >= MAX_CHAMADAS:
            return [Acao.SAIR] if self.pos == INICIO else self._voltar()
        estado = self._texto_estado(p)
        for _ in range(2):  # uma nova tentativa se a resposta vier inválida
            texto = self._perguntar(estado)
            plano = self._interpretar(texto, p)
            if plano is not None:
                return plano
            self.respostas_invalidas += 1
        return [Acao.SAIR] if self.pos == INICIO else self._voltar()

    def _voltar(self) -> list:
        destino = self.anterior if self.anterior and self.anterior in vizinhos(self.pos) else None
        destino = destino or min((v for v in vizinhos(self.pos) if v in self.observacoes),
                                 key=lambda v: v[0] + v[1], default=None)
        return acoes_para_vizinho(self.pos, self.direcao, destino) if destino else [Acao.VIRAR_ESQUERDA]

    def _perguntar(self, estado: str) -> str:
        inicio = time.perf_counter()
        texto, t_in, t_out = self.cliente(SISTEMA, estado, self.modelo)
        self.tempo_llm += time.perf_counter() - inicio
        self.chamadas += 1
        self.tokens_entrada += t_in
        self.tokens_saida += t_out
        if self.log:
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"estado": estado, "resposta": texto}, ensure_ascii=False) + "\n")
        return texto

    def _interpretar(self, texto: str, p: Percepcao):
        m = re.search(r"\{.*\}", texto, re.S)
        if not m:
            return None
        try:
            r = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

        self._auditar(r)

        acao = str(r.get("acao", "")).upper()
        if acao == "AGARRAR":
            return [Acao.AGARRAR]
        if acao == "SAIR":
            return [Acao.SAIR]  # fora de (1,1) não faz nada: o erro custa −1, como no jogo
        if acao == "ATIRAR":
            try:
                alvo = Direcao[str(r.get("direcao", "")).upper()]
            except KeyError:
                return None
            giros = acoes_para_vizinho((2, 2), self.direcao,
                                       (2 + alvo.delta[0], 2 + alvo.delta[1]))[:-1]
            return giros + [Acao.ATIRAR]
        if acao == "MOVER":
            destino = _casa(r.get("destino"))
            if destino is None or destino not in vizinhos(self.pos):
                return None
            self.movimentos += 1
            if destino not in self.observacoes and not self.sombra.ask(sentenca_segura(destino)):
                self.movimentos_inseguros += 1
            return acoes_para_vizinho(self.pos, self.direcao, destino)
        return None

    def _auditar(self, r: dict) -> None:
        """Compara o que o LLM AFIRMA com o que a lógica consegue provar."""
        for campo, disse_segura in (("casas_seguras", True), ("casas_perigosas", False)):
            itens = r.get(campo) or []
            if not isinstance(itens, list):
                continue
            for v in itens:
                c = _casa(v)
                if c is None or c in self.observacoes:
                    continue
                self.afirmacoes += 1
                segura = self.sombra.ask(sentenca_segura(c))
                perigosa = self.sombra.ask(P(c[0], c[1])) or (
                    self.sombra.ask(W(c[0], c[1])) and not self.sombra.ask(M))
                if disse_segura and perigosa or (not disse_segura and segura):
                    self.alucinacoes += 1
                    if len(self.exemplos_alucinacao) < 3:
                        self.exemplos_alucinacao.append(
                            f"{c} chamada de {'segura' if disse_segura else 'perigosa'}; "
                            f"a BC prova o contrário. Raciocínio: {str(r.get('raciocinio'))[:200]}")
                elif disse_segura and not segura:
                    self.sem_prova += 1

    def depurar(self) -> str:
        return (f"chamadas LLM: {self.chamadas} | tokens: {self.tokens_entrada}+{self.tokens_saida} | "
                f"inseguros: {self.movimentos_inseguros}/{self.movimentos} | "
                f"alucinações: {self.alucinacoes}/{self.afirmacoes}")

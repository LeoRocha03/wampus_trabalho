"""AGENTE A — Reativo simples (Aula 4, tipo 1).                      [0,3 ponto]

Regras condição → ação, SEM MEMÓRIA: a decisão depende apenas da percepção atual.
Nada é guardado em self entre chamadas (nem "já peguei o ouro", nem casas
visitadas, nem se a flecha já foi usada).

Respostas às perguntas do esqueleto
-----------------------------------
- Brilho → AGARRAR. É a única regra que leva ao ouro.
- Sem memória o agente NÃO sabe se já está com o ouro. Por isso a regra de saída
  não pode depender disso: ele sai sempre que está em (1,1) voltando para casa
  (virado para oeste ou sul, ou seja, chegou por uma casa vizinha). Se por acaso
  agarrou o ouro antes, ganha +1000; se não, sai com o que tem. Esse é o limite
  estrutural da arquitetura, e aparece no experimento.
- Brisa ou fedor → perigo numa casa vizinha, mas sem memória não dá para saber
  de onde veio. A regra é girar na maioria das vezes em vez de avançar.
- Fedor → atirar para a frente. Sem memória, ele não sabe se ainda tem a flecha;
  atirar sem flecha só custa −1, então a regra é aceitável.
- Baque → girar.
- Como evitar girar para sempre? Um agente reativo DETERMINÍSTICO fica preso em
  laços (mesma percepção, mesma ação). Usamos o agente reativo ALEATORIZADO de
  Russell & Norvig (seção 2.4.2): a regra sorteia entre ações. O gerador fica no
  módulo, não em self, e não guarda nada sobre o mundo — só produz sorteios.

A percepção inclui posição e direção (simplificação didática do simulador); as
regras usam apenas a percepção do instante atual.
"""

import random

from wumpus import Acao, Agente, Direcao, Percepcao

_sorteio = random.Random(2026)


class AgenteReativo(Agente):
    nome = "A · Reativo"

    def agir(self, percepcao: Percepcao) -> Acao:
        p = percepcao

        # R1: ouro aqui → agarrar
        if p.brilho:
            return Acao.AGARRAR

        # R2: em (1,1) voltando (virado para oeste/sul) → sair
        if p.posicao == (1, 1) and p.direcao in (Direcao.OESTE, Direcao.SUL):
            return Acao.SAIR

        # R3: bateu na parede → girar
        if p.baque:
            return _sorteio.choice([Acao.VIRAR_ESQUERDA, Acao.VIRAR_DIREITA])

        # R4: fedor → atirar para a frente (metade das vezes)
        if p.fedor and _sorteio.random() < 0.5:
            return Acao.ATIRAR

        # R5: brisa ou fedor → preferir girar a avançar
        if p.brisa or p.fedor:
            return _sorteio.choices(
                [Acao.VIRAR_ESQUERDA, Acao.VIRAR_DIREITA, Acao.AVANCAR],
                weights=[0.4, 0.4, 0.2])[0]

        # R6: nada percebido → avançar na maioria das vezes
        return _sorteio.choices(
            [Acao.AVANCAR, Acao.VIRAR_ESQUERDA, Acao.VIRAR_DIREITA],
            weights=[0.7, 0.15, 0.15])[0]

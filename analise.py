#!/usr/bin/env python3
"""Métricas extras para o relatório (NÃO é usado pelos agentes).

Este script lê o estado real de cada mundo (posição do ouro, poços, Wumpus) para
CLASSIFICAR os mundos depois da partida. Os agentes nunca veem essas informações.

  python analise.py                 # sementes públicas 0..199
  python analise.py -n 1000 --semente-base 1000
"""

import argparse
import statistics
from collections import Counter, deque

from wumpus import MundoWumpus, rodar_episodio, vizinhos
from agentes.agente_reativo import AgenteReativo
from agentes.agente_logico import AgenteLogico
from agentes.agente_objetivo import AgenteObjetivo


def classificar(m: MundoWumpus) -> str:
    """Categoria do mundo, do ponto de vista de um agente com informação perfeita."""
    if m.ouro in m.pocos:
        return "ouro no poço"
    # caminho sem poços até o ouro (o Wumpus pode ser morto com a flecha)
    livres = {(x, y) for x in range(1, 5) for y in range(1, 5)} - set(m.pocos)
    fila, vistos = deque([(1, 1)]), {(1, 1)}
    while fila:
        c = fila.popleft()
        for v in vizinhos(c):
            if v in livres and v not in vistos:
                vistos.add(v)
                fila.append(v)
    if m.ouro not in vistos:
        return "ouro isolado por poços"
    inicio = m.percepcao()
    if inicio.brisa or inicio.fedor:
        return "solucionável, alarme em (1,1)"
    return "solucionável, (1,1) calmo"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--semente-base", type=int, default=0)
    a = p.parse_args()
    sementes = range(a.semente_base, a.semente_base + a.n)

    categorias = {s: classificar(MundoWumpus.gerar(s)) for s in sementes}
    print("## Mundos por categoria\n")
    for cat, n in Counter(categorias.values()).most_common():
        print(f"- {cat}: {n} ({100 * n / a.n:.1f}%)")

    agentes = {"A": AgenteReativo, "B": AgenteLogico, "C": AgenteObjetivo}
    print("\n## Sucesso por categoria de mundo\n")
    print("| Categoria | n | " + " | ".join(agentes) + " |")
    print("|---|---:|" + "---:|" * len(agentes))
    resultados = {k: {} for k in agentes}
    instancias = {k: {} for k in agentes}
    for k, cls in agentes.items():
        for s in sementes:
            caixa = []
            def fabrica(cls=cls, caixa=caixa):
                caixa.append(cls())
                return caixa[-1]
            resultados[k][s] = rodar_episodio(fabrica, s)
            instancias[k][s] = caixa[-1]
    for cat in sorted(set(categorias.values())):
        ss = [s for s in sementes if categorias[s] == cat]
        linha = [f"{100 * sum(resultados[k][s].sucesso for s in ss) / len(ss):.1f}%" for k in agentes]
        print(f"| {cat} | {len(ss)} | " + " | ".join(linha) + " |")

    print("\n## Mortes por categoria\n")
    for k in agentes:
        c = Counter(categorias[s] for s in sementes if resultados[k][s].morte)
        print(f"- {k}: {dict(c)}")

    print("\n## Custo da inferência (por partida, média)\n")
    for k in ("B", "C"):
        ags = list(instancias[k].values())
        bcs = [g.bc for g in ags]
        cons = statistics.mean(b.consultas for b in bcs)
        res = statistics.mean(b.consultas_resolvidas for b in bcs)
        rsv = statistics.mean(b.estat.resolventes for b in bcs)
        est = sum(b.estat.orcamento_estourado for b in bcs)
        cla = statistics.mean(len(b) for b in bcs)
        simb = statistics.mean(len({s for c in b.clausulas for s, _ in c}) for b in bcs)
        tempo = statistics.mean(r.tempo_s for r in resultados[k].values()) * 1000
        print(f"- {k}: ask()={cons:.1f}, refutações executadas (fora do cache)={res:.1f}, "
              f"resolventes gerados={rsv:.0f}, orçamento estourado={est}, "
              f"cláusulas finais={cla:.0f}, símbolos na BC={simb:.1f} "
              f"(tabela-verdade: 2^{simb:.0f} modelos), tempo={tempo:.2f} ms")
    c_ags = list(instancias["C"].values())
    print(f"- C: nós BFS={statistics.mean(g.nos_expandidos for g in c_ags):.1f}, "
          f"partidas com risco assumido={sum(g.riscos_assumidos > 0 for g in c_ags)}, "
          f"flechas usadas={sum(not g.tem_flecha for g in c_ags)}")
    mortes_c = [s for s in sementes if resultados['C'][s].morte]
    print(f"- C: mortes, todas em fase 'arriscar'? "
          f"{all(instancias['C'][s].fase == 'arriscar' for s in mortes_c)} ({len(mortes_c)} mortes)")


if __name__ == "__main__":
    main()

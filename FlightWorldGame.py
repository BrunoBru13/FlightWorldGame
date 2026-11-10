"""FlightWorld - simulatore di volo 3D, Python + Panda3D.

Sei aerei (Cessna 172, Boeing 737-800, Airbus A320, Airbus A380, Concorde, jet privato) e
otto grandi aeroporti reali (Roma, New York, Milano, Londra, Parigi, Dubai, Los Angeles, Tokyo)
su un'isola generata proceduralmente, con grattacieli, la metropoli col Burj Khalifa,
passerelle d'imbarco con i passeggeri, schianti realistici, multiplayer e l'hangar dove
personalizzare colori, livree delle compagnie, scritte, motore e peso degli aerei.
Il mouse indica dove vuoi andare (destra/sinistra = virata, alto/basso = salita
o discesa); E motori, W acceleratore, S freni, P retromarcia, G carrello, R quota fissa.

Avvio:  python flightsim.py
"""
import argparse
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave

ap_ = argparse.ArgumentParser()
ap_.add_argument('--selftest', action='store_true', help='prova automatica offscreen')
ap_.add_argument('--aircraft', type=int, default=0)
ap_.add_argument('--hour', type=float, default=-1.0, help='ora di partenza (0-24); di default le 8 del mattino')
ap_.add_argument('--airport', type=int, default=0)
ap_.add_argument('--air', action='store_true')
ap_.add_argument('--out', default='.')
ap_.add_argument('--scene', default='', help='prova di una scena: hangar, citta, burj, schianto, acqua, edificio')
ARGS = ap_.parse_args()

from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-title FlightWorld')
if ARGS.selftest:
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
elif os.environ.get('FW_LOADSHOT'):
    loadPrcFileData('', 'window-type offscreen\nwin-size 1280 720\naudio-library-name null')
else:
    loadPrcFileData('', 'win-size 1280 720\nsync-video 1\nframebuffer-multisample 1\nmultisamples 4')

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageChops
except ImportError:
    raise SystemExit('Servono numpy e pillow:  pip install numpy pillow')
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectFrame, DirectButton, DirectEntry, DGG
from direct.interval.IntervalGlobal import Sequence, Parallel, Wait, LerpPosInterval, LerpScaleInterval, LerpColorScaleInterval, Func
from panda3d.core import (AmbientLight, AntialiasAttrib, ButtonRegistry, CardMaker, ColorBlendAttrib,
                          DirectionalLight, Filename, Fog, PointLight, Geom, GeomEnums, GeomNode, GeomTriangles, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, LineSegs, Mat3, NodePath, PNMImage,
                          GeomPoints, GeomVertexArrayFormat, InternalName, Camera, PerspectiveLens, WindowProperties, OrthographicLens, GeomVertexFormat as _GVF,
                          Point3, Quat, SamplerState, TextNode, Texture, TransparencyAttrib, Vec3)


# ---------------------------------------------------------------- matematica
def clamp(x, a, b):
    return a if x < a else b if x > b else x


def smoothstep(a, b, x):
    t = clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _hash(ix, iy, seed):
    n = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    n ^= n >> 16
    return n / 4294967295.0


def vnoise(x, y, seed=0):
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    a = _hash(ix, iy, seed)
    b = _hash(ix + 1, iy, seed)
    c = _hash(ix, iy + 1, seed)
    d = _hash(ix + 1, iy + 1, seed)
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy


def fbm(x, y, octaves=5, seed=0):
    s = tot = 0.0
    amp = 0.5
    for i in range(octaves):
        s += amp * vnoise(x, y, seed + i)
        tot += amp
        x = x * 2.03 + 17.1
        y = y * 2.03 - 9.3
        amp *= 0.5
    return s / tot


def rot(v, k, ang):
    """Ruota v attorno all'asse unitario k di ang radianti."""
    c, s = math.cos(ang), math.sin(ang)
    return v * c + k.cross(v) * s + k * (k.dot(v) * (1 - c))


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------------------------------------------------------------- aeroporti
GATE_U = 300.0                                          # facciata del terminal (coordinate locali dell'aeroporto)
GATE_V = (-400.0, -240.0, -80.0, 80.0, 240.0, 400.0)    # posizione lungo la pista delle sei passerelle
CARGO_V = (1130.0, 1330.0)                              # i due stalli del terminal cargo (nastri di carico)
GRID_N = 300                                            # mappa piu grande: 120 km x 120 km
GRID_C = 400.0
MAP_R = 60000.0
GRID_X0 = -GRID_N * GRID_C / 2


class Airport(Obj):
    def __init__(self, code, name, cx, cy, h, hdg, L, labs=('', ''), extra=(), W=60.0, th=520.0, feat=(),
                 sea=None, rock=None, note=''):
        # sea: None | ('island',) | ('ends', metri_in_mare_al_capo_-v, metri_in_mare_al_capo_+v)
        # rock: (u, v, raggio, altezza) roccia o montagna accanto all'aeroporto
        Obj.__init__(self, code=code, name=name, cx=cx, cy=cy, h=h, hdg=hdg, L=L, W=W, th=th, feat=feat,
                     sea=sea, rock=rock, note=note)
        # piste: la principale (u=0) piu quelle reali dell'aeroporto: (u, v, angolo orario, lunghezza, larghezza, nomi)
        # piantina: due piste parallele MOLTO vicine (distanza fissa tra i bordi), raccordi a X in mezzo,
        # tre raccordi verso il lato terminal, una fascia di rullaggio e poi il piazzale con i gate
        ex2 = []
        for (u, v, ang, ln, wd, lb) in extra:
            if abs(ang) < 0.1 and abs(u) > 150.0:
                u = -(W / 2 + wd / 2 + 105.0)
            ex2.append((u, v, ang, ln, wd, lb))
        extra = ex2
        self.rws = []
        for (u, v, ang, ln, wd, lb) in [(0.0, 0.0, 0.0, L, W, labs)] + list(extra):
            r = dict(u=u, v=v, ang=ang, L=ln, W=wd, labs=lb)
            r['c'], r['s'] = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            self.rws.append(r)
        umax, vmax = 440.0, max(L / 2 + 440.0, 1620.0)   # l'area cargo arriva a v = 1440
        self.pave = [(90.0, 290.0, 700.0, 1500.0),       # il piazzale si allunga fino all'area cargo
                     (90.0, 125.0, -L / 2 + 150.0, L / 2 - 150.0),     # fascia di rullaggio lungo il piazzale
                     (W / 2, 90.0, L / 2 - 275.0, L / 2 - 225.0)]      # terzo raccordo (gli altri due sono in build_airports)
        self.taxi = []                       # raccordi in obliquo: (u0, v0, u1, v1, larghezza)
        for r in self.rws:
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                x, y = sx * r['W'] / 2, sy * r['L'] / 2
                umax = max(umax, abs(r['u'] + x * r['c'] + y * r['s']) + 300.0)
                vmax = max(vmax, abs(r['v'] - x * r['s'] + y * r['c']) + 440.0)
        for r in self.rws[1:]:
            if abs(r['ang']) < 0.1 and abs(r['u']) > 150.0:
                ue = r['u']
                Lm = min(L, r['L']) / 2 - 150.0
                vv = Lm - 100.0
                for sgn in (-1, 1):             # raccordi perpendicolari in testata
                    self.pave.append((min(0.0, ue), max(0.0, ue), sgn * vv - 25.0, sgn * vv + 25.0))
                ua, ub = -W / 2 + 26.0, ue + r['W'] / 2 - 26.0     # i raccordi entrano nelle piste: nessun vuoto agli attacchi
                dvs = abs(ue) * 1.3 + 40.0
                for f0 in (-0.6, 0.0, 0.6):                         # zig-zag come nella piantina: ogni diagonale parte dove finisce la precedente
                    v = f0 * Lm - 3 * dvs
                    for k in range(6):
                        pa, pb = (ua, ub) if k % 2 == 0 else (ub, ua)
                        self.taxi.append((pa, v, pb, v + dvs, 28.0))
                        v += dvs
        self.hx = umax if hdg == 'N' else vmax
        self.hy = vmax if hdg == 'N' else umax

    def rw_xy(self, r, u, v):
        """Coordinate locali (u, v) -> coordinate della pista r (x trasversale, y lungo la pista)."""
        du, dv = u - r['u'], v - r['v']
        return du * r['c'] - dv * r['s'], du * r['s'] + dv * r['c']

    def local(self, x, y):
        return (x - self.cx, y - self.cy) if self.hdg == 'N' else (-(y - self.cy), x - self.cx)

    def world(self, u, v):
        return (self.cx + u, self.cy + v) if self.hdg == 'N' else (self.cx + v, self.cy - u)

    def fwd(self):
        return Vec3(0, 1, 0) if self.hdg == 'N' else Vec3(1, 0, 0)


# aeroporti famosi (nomi e lunghezze delle piste principali reali); le piste sono orientate in modo da non
# puntare contro montagne o grattacieli, e il terreno attorno alle rotte di avvicinamento viene spianato
# piste extra: (u, v, angolo orario rispetto alla principale, lunghezza, larghezza, (nome verso +v, nome verso -v))
# Ogni aeroporto reale ha le sue due piste parallele (distanze e lunghezze prese dagli aeroporti veri)
# collegate da raccordi obliqui; quelli "folli" hanno piste sul mare, su pilastri o su un'isola.
APTS = [
    Airport('FCO', 'Roma Fiumicino', 0, 0, 20, 'N', 3900, ('34L', '16R'), [(-1000, 0, 0, 3680, 45, ('34R', '16L'))]),
    Airport('JFK', 'New York JFK', 28000, 8000, 8, 'E', 4423, ('13L', '31R'), [(-900, 0, 0, 3048, 45, ('13R', '31L'))]),
    Airport('MXP', 'Milano Malpensa', -24000, 22000, 140, 'N', 3920, ('35R', '17L'), [(-800, 0, 0, 3920, 60, ('35L', '17R'))]),
    Airport('LHR', 'Londra Heathrow', 4000, 36000, 35, 'E', 3902, ('09L', '27R'), [(-1400, 0, 0, 3660, 50, ('09R', '27L'))],
            th=700),
    Airport('CDG', 'Parigi Charles de Gaulle', -8000, -32000, 12, 'E', 4200, ('09R', '27L'),
            [(-400, 0, 0, 2700, 45, ('09L', '27R'))], th=650, feat=(('rot', 800.0, -250.0, 75.0, 28.0),)),
    Airport('DXB', 'Dubai Intl', 22500, 18000, 14, 'E', 4447, ('12L', '30R'), [(-385, 0, 0, 4000, 60, ('12R', '30L'))], th=780),
    Airport('LAX', 'Los Angeles Intl', -34000, -6000, 38, 'E', 3685, ('07L', '25R'), [(-300, 0, 0, 3380, 45, ('07R', '25L'))],
            th=620),
    Airport('HND', 'Tokyo Haneda', 31000, -14000, 6, 'E', 3360, ('34R', '16L'), [(-900, 0, 0, 3000, 60, ('34L', '16R'))]),
    # --- aeroporti "folli", ai bordi della mappa
    Airport('GIB', 'Gibraltar (pista nel mare)', -50000, 14000, 5, 'E', 1900, ('09', '27'), [], th=450,
            sea=('ends', 520.0, 520.0), rock=(2700.0, 0.0, 800.0, 420.0), note='Pista corta con tutti e due i capi in mare\ne la Rocca accanto.'),
    Airport('SXM', 'Saint Martin (spiaggia)', 30000, -49000, 4, 'E', 2300, ('10', '28'), [], th=450,
            sea=('ends', 80.0, 0.0), rock=(-2000.0, 600.0, 600.0, 300.0), note='Si atterra radenti alla spiaggia:\nil mare finisce a un passo dalla pista.'),
    Airport('FNC', 'Madeira Funchal (pilastri)', -30000, 47000, 58, 'N', 2800, ('05', '23'), [], th=450,
            sea=('ends', 0.0, 1000.0), rock=(-2300.0, -600.0, 900.0, 650.0), note='Un tratto di pista di 1 km e sospeso\nsul mare, su alti pilastri.'),
    Airport('HKG', 'Hong Kong (isola)', 50000, 30000, 6, 'E', 3800, ('07L', '25R'), [(-1500, 0, 0, 3800, 60, ('07R', '25L'))],
            th=600, sea=('island',), note='Aeroporto su un\'isola artificiale\ncon due piste parallele.')]
APPROACH_SLOPE = 0.03                     # il terreno non sale mai oltre il 3% lungo l'asse di pista

# tratti di pista sospesi sul mare: (x0, x1, y0, y1, quota) in coordinate del mondo; il terreno li vede come suolo piatto
PIERS = []
for _a in APTS:
    if _a.sea and _a.sea[0] == 'ends':
        _hw = _a.W / 2 + 40.0
        for _end, _ext in ((-1, _a.sea[1]), (1, _a.sea[2])):
            if _ext > 0:
                _v0, _v1 = (_a.L / 2 - _ext, _a.L / 2 + 30.0) if _end > 0 else (-_a.L / 2 - 30.0, -_a.L / 2 + _ext)
                _c = [_a.world(_u, _v) for _u in (-_hw, _hw) for _v in (_v0, _v1)]
                _a.__dict__.setdefault('piers', []).append((_v0, _v1))
                PIERS.append((min(p[0] for p in _c), max(p[0] for p in _c), min(p[1] for p in _c), max(p[1] for p in _c), _a.h))


# ostacoli (grattacieli, torri): cerchi (x, y, raggio, quota assoluta della cima) in una griglia spaziale
OBS_CELL = 500.0
OBS_GRID = {}


def add_obstacle(x, y, r, top):
    for i in range(int((x - r) // OBS_CELL), int((x + r) // OBS_CELL) + 1):
        for j in range(int((y - r) // OBS_CELL), int((y + r) // OBS_CELL) + 1):
            OBS_GRID.setdefault((i, j), []).append((x, y, r, top))


# la metropoli moderna (con il Burj Khalifa): terreno piatto attorno al centro
CITY_X, CITY_Y, CITY_H = 22500.0, 13500.0, 14.0
CITY_CORE, CITY_FADE = 1800.0, 3200.0


def near_city(x, y, m=0.0):
    return math.hypot(x - CITY_X, y - CITY_Y) < CITY_FADE + m


def near_airport(x, y, m=0.0):
    return any(abs(x - a.cx) < a.hx + m and abs(y - a.cy) < a.hy + m for a in APTS)


def in_corridor(x, y, lat=2300.0, far=12500.0):
    """Nel corridoio di avvicinamento di una qualsiasi pista (li non si costruiscono grattacieli)."""
    for a in APTS:
        if abs(x - a.cx) > 30000.0 or abs(y - a.cy) > 30000.0:
            continue
        u0, v0 = a.local(x, y)
        for r in a.rws:
            xr, yr = a.rw_xy(r, u0, v0)
            if abs(xr) < lat and r['L'] / 2 < abs(yr) < r['L'] / 2 + far:
                return True
    return False


def on_airport(x, y):
    """Dentro l'area di un aeroporto (pista, raccordi, prato attorno)."""
    return any(abs(x - a.cx) < a.hx and abs(y - a.cy) < a.hy for a in APTS)


def _rw_T(r):
    """Da coordinate pista (x trasversale, y lungo) a coordinate locali dell'aeroporto (u, v)."""
    c, s, u0, v0 = r['c'], r['s'], r['u'], r['v']
    return lambda x, y: (u0 + x * c + y * s, v0 - x * s + y * c)


_SEG = {'a': (0, 4, 7.1, 8), 'd': (0, 4, 0, 0.9), 'g': (0, 4, 3.55, 4.45), 'f': (0, 0.9, 4, 8),
        'b': (3.1, 4, 4, 8), 'e': (0, 0.9, 0, 4), 'c': (3.1, 4, 0, 4)}
_GLYPH = {'0': 'abcdef', '1': 'bc', '2': 'abged', '3': 'abgcd', '4': 'fgbc', '5': 'afgcd', '6': 'afgedc',
          '7': 'abc', '8': 'abcdefg', '9': 'abcdfg', 'L': 'fed', 'R': 'abfge', 'C': 'afed'}


def _rw_label(mk, fr, text, y0, flip):
    """Scrive il numero di pista (es. 34L) con segmenti, nel senso di marcia di chi atterra."""
    step = 6.0
    total = len(text) * step - 2.0
    for i, ch in enumerate(text):
        gx = -total / 2 + i * step
        rects = [_SEG[s] for s in _GLYPH.get(ch, '')]
        if ch == 'R':                                        # gamba della R
            rects += [(1.8, 2.7, 2.2, 3.55), (2.6, 3.5, 0, 2.2)]
        for (x0, x1, q0, q1) in rects:
            a0, a1, b0, b1 = gx + x0, gx + x1, y0 + q0, y0 + q1
            if flip:
                a0, a1, b0, b1 = -a1, -a0, -b1, -b0
            fr(a0, a1, b0, b1, (0.93, 0.93, 0.90, 1), 0.14)


def on_pavement(x, y):
    for a in APTS:
        if abs(x - a.cx) < a.hx and abs(y - a.cy) < a.hy:
            u, v = a.local(x, y)
            for r in a.rws:
                xr, yr = a.rw_xy(r, u, v)
                if abs(xr) <= r['W'] / 2 and abs(yr) <= r['L'] / 2:
                    return True
            if any(p0 <= u <= p1 and q0 <= v <= q1 for (p0, p1, q0, q1) in a.pave):
                return True
            for (u0, v0, u1, v1, w) in a.taxi:
                du, dv = u1 - u0, v1 - v0
                t = clamp(((u - u0) * du + (v - v0) * dv) / (du * du + dv * dv), 0.0, 1.0)
                if math.hypot(u - u0 - du * t, v - v0 - dv * t) <= w / 2:
                    return True
            if 90 <= u <= 290 and abs(v) <= 700:
                return True
            if a.W / 2 <= u <= 90 and (abs(v) <= 25 or abs(v + a.L / 2 - 250) <= 25):
                return True
    return False


# ---------------------------------------------------------------- terreno
def terrain_raw(x, y):
    dmin = min(math.hypot(x - a.cx, y - a.cy) for a in APTS)
    relief = 0.2 + 0.8 * smoothstep(2000, 8000, dmin)
    n = fbm(x / 9000.0 + 13.7, y / 9000.0 - 4.2, 5)
    e = (n - 0.45) * 2800.0 * relief + 150.0 * (1 - smoothstep(0, 5000, dmin))
    r = math.hypot(x, y)
    mask = 1 - smoothstep(MAP_R - 20000.0, MAP_R - 12000.0, r)
    e = e * mask - 400.0 * (1 - mask)
    for a in APTS:
        dx = max(abs(x - a.cx) - a.hx, 0.0)
        dy = max(abs(y - a.cy) - a.hy, 0.0)
        if dx < 1500 and dy < 1500:
            t = smoothstep(0, 1500, math.hypot(dx, dy))
            e = a.h * (1 - t) + e * t
        # corridoio di avvicinamento: lungo l'asse della pista il terreno e tenuto basso (nessuna montagna davanti)
        if abs(x - a.cx) > 28000.0 or abs(y - a.cy) > 28000.0:
            continue
        u, v = a.local(x, y)
        if a.sea:
            if a.sea[0] == 'island':                           # isola artificiale: tutt'intorno e mare
                dd = math.hypot(dx, dy)
                if dd < 9000.0:
                    se = a.h * (1 - smoothstep(100.0, 800.0, dd)) - 16.0 * smoothstep(100.0, 800.0, dd)
                    w = 1.0 - smoothstep(5000.0, 9000.0, dd)
                    e = e * (1 - w) + se * w
            else:                                              # capi pista in mare
                ds = max(v - (a.L / 2 - a.sea[2]) if a.sea[2] > 0 else -1e9,
                         (-a.L / 2 + a.sea[1]) - v if a.sea[1] > 0 else -1e9)
                if ds > -1e8 and ds < 6500.0 and abs(u) < 4000.0:
                    w = smoothstep(0.0, 500.0, ds) * (1.0 - smoothstep(4000.0, 6500.0, ds)) * (1.0 - smoothstep(3000.0, 4000.0, abs(u)))
                    w *= 1.0 - smoothstep(60.0, 70.0, u)       # sul lato del terminal resta terra (riva al bordo del pontile)
                    e = e * (1 - w) + (-16.0) * w
        if a.rock:                                             # roccia/montagna accanto all'aeroporto
            ru, rv, rr, rh = a.rock
            d2 = ((u - ru) ** 2 + (v - rv) ** 2) / (rr * rr)
            if d2 < 4.0:                                       # dolce, e si spegne a due raggi dal centro
                e = max(e, a.h + rh * (1.0 - d2 / 4.0) ** 2)
        for r in a.rws:
            xr, yr = a.rw_xy(r, u, v)
            dv = abs(yr) - r['L'] / 2
            if dv > 0.0 and dv < 18000.0 and abs(xr) < 2600.0:
                w = 1.0 - smoothstep(1100.0, 2600.0, abs(xr))
                cap = a.h + APPROACH_SLOPE * dv + 6.0
                if e > cap:
                    e = e * (1 - w) + cap * w
    dc = math.hypot(x - CITY_X, y - CITY_Y)
    if dc < CITY_FADE:
        t = smoothstep(CITY_CORE, CITY_FADE, dc)
        e = CITY_H * (1 - t) + e * t
    return e


_LD = [None]


def ld_pump(prog=None):
    """Fa avanzare l'animazione di caricamento dai cicli lunghi (se e attiva)."""
    if _LD[0] is not None:
        _LD[0].pump(prog)


class Terrain:
    """Griglia di quote: lo stesso reticolo disegna la mesh e serve alle collisioni."""

    def __init__(self):
        n = GRID_N + 1
        self.H = []
        for j in range(n):
            self.H.extend(terrain_raw(GRID_X0 + i * GRID_C, GRID_X0 + j * GRID_C) for i in range(n))
            if j % 3 == 0:
                ld_pump(0.02 + 0.40 * j / n)

    def sample(self, x, y):
        """Ritorna (quota fisica >= 0, quota grezza: negativa = mare)."""
        n = GRID_N + 1
        for p in PIERS:
            if p[0] <= x <= p[1] and p[2] <= y <= p[3]:
                return p[4], p[4]
        gx = (x - GRID_X0) / GRID_C
        gy = (y - GRID_X0) / GRID_C
        i = math.floor(gx)
        j = math.floor(gy)
        if i < 0 or j < 0 or i >= GRID_N or j >= GRID_N:
            raw = terrain_raw(x, y)
            return max(raw, 0.0), raw
        fx = gx - i
        fy = gy - j
        H = self.H
        h00 = H[j * n + i]
        h10 = H[j * n + i + 1]
        h01 = H[(j + 1) * n + i]
        h11 = H[(j + 1) * n + i + 1]
        if fx >= fy:
            raw = h00 + (h10 - h00) * fx + (h11 - h10) * fy
        else:
            raw = h00 + (h11 - h01) * fx + (h01 - h00) * fy
        return max(raw, 0.0), raw


# ---------------------------------------------------------------- mesh
class Mesh:
    def __init__(self):
        self.P, self.N, self.C, self.T = [], [], [], []

    def vert(self, p, n, c):
        self.P.append(p)
        self.N.append(n)
        self.C.append(c)
        return len(self.P) - 1

    def quad(self, pts, col, center):
        p0, p1, p2, p3 = [Vec3(*p) for p in pts]
        n = (p1 - p0).cross(p2 - p0)
        if n.dot((p0 + p1 + p2 + p3) * 0.25 - Vec3(*center)) < 0:
            p1, p3 = p3, p1
            n = -n
        n.normalize()
        nt = (n.x, n.y, n.z)
        i = [self.vert((p.x, p.y, p.z), nt, col) for p in (p0, p1, p2, p3)]
        self.T.append((i[0], i[1], i[2]))
        self.T.append((i[0], i[2], i[3]))

    def box(self, x0, x1, y0, y1, z0, z1, col):
        c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
        X, Y, Z = (x0, x1), (y0, y1), (z0, z1)
        for k in (0, 1):
            self.quad([(X[k], Y[0], Z[0]), (X[k], Y[1], Z[0]), (X[k], Y[1], Z[1]), (X[k], Y[0], Z[1])], col, c)
            self.quad([(X[0], Y[k], Z[0]), (X[1], Y[k], Z[0]), (X[1], Y[k], Z[1]), (X[0], Y[k], Z[1])], col, c)
            self.quad([(X[0], Y[0], Z[k]), (X[1], Y[0], Z[k]), (X[1], Y[1], Z[k]), (X[0], Y[1], Z[k])], col, c)

    def hull(self, p, col):
        """Solido convesso: 4 punti in basso e 4 in alto, nello stesso ordine."""
        c = tuple(sum(q[i] for q in p) / 8.0 for i in range(3))
        for f in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            self.quad([p[i] for i in f], col, c)

    def tri_flat(self, p0, p1, p2, col, center):
        p0, p1, p2 = Vec3(*p0), Vec3(*p1), Vec3(*p2)
        n = (p1 - p0).cross(p2 - p0)
        if n.dot((p0 + p1 + p2) / 3.0 - Vec3(*center)) < 0:
            p1, p2 = p2, p1
            n = -n
        n.normalize()
        nt = (n.x, n.y, n.z)
        i = [self.vert((p.x, p.y, p.z), nt, col) for p in (p0, p1, p2)]
        self.T.append(tuple(i))

    def _oriented(self, a, b, c):
        pa, pb, pc = Vec3(*self.P[a]), Vec3(*self.P[b]), Vec3(*self.P[c])
        n = (pb - pa).cross(pc - pa)
        if n.dot(Vec3(*self.N[a])) < 0:
            b, c = c, b
        self.T.append((a, b, c))

    def loft(self, stations, sides, col, colfn=None):
        """Corpo di rivoluzione lungo Y: stazioni (y, rx, rz, zc); colfn(angolo, indice) colora per vertice."""
        rings = []
        for k, (y, rx, rz, zc) in enumerate(stations):
            ring = []
            for s in range(sides):
                a = 2 * math.pi * s / sides
                ca, sa = math.cos(a), math.sin(a)
                ring.append(self.vert((rx * ca, y, zc + rz * sa), (ca, 0.0, sa), colfn(a, k) if colfn else col))
            rings.append(ring)
        for k in range(len(rings) - 1):
            for s in range(sides):
                s2 = (s + 1) % sides
                a, b = rings[k][s], rings[k][s2]
                c, d = rings[k + 1][s2], rings[k + 1][s]
                self._oriented(a, b, c)
                self._oriented(a, c, d)

    def sphere(self, cx, cy, cz, rx, ry, rz, col, nlon=8, nlat=5):
        rings = []
        for la in range(nlat + 1):
            th = math.pi * la / nlat
            rr, z = math.sin(th), math.cos(th)
            ring = []
            for lo in range(nlon):
                ph = 2 * math.pi * lo / nlon
                cp, sp = math.cos(ph), math.sin(ph)
                ring.append(self.vert((cx + rr * rx * cp, cy + rr * ry * sp, cz + z * rz),
                                      (rr * cp, rr * sp, z), col))
            rings.append(ring)
        for k in range(nlat):
            for s in range(nlon):
                s2 = (s + 1) % nlon
                a, b = rings[k][s], rings[k][s2]
                c, d = rings[k + 1][s2], rings[k + 1][s]
                self._oriented(a, b, c)
                self._oriented(a, c, d)

    def cone(self, cx, cy, cz, radius, height, sides, col):
        apex = (cx, cy, cz + height)
        axis = (cx, cy, cz + height * 0.4)
        for s in range(sides):
            a0 = 2 * math.pi * s / sides
            a1 = 2 * math.pi * (s + 1) / sides
            b0 = (cx + radius * math.cos(a0), cy + radius * math.sin(a0), cz)
            b1 = (cx + radius * math.cos(a1), cy + radius * math.sin(a1), cz)
            self.tri_flat(b0, b1, apex, col, axis)

    def node(self, name):
        vd = GeomVertexData(name, GeomVertexFormat.getV3n3c4(), Geom.UHStatic)
        vd.setNumRows(len(self.P))
        vw = GeomVertexWriter(vd, 'vertex')
        nw = GeomVertexWriter(vd, 'normal')
        cw = GeomVertexWriter(vd, 'color')
        for p, n, c in zip(self.P, self.N, self.C):
            vw.addData3(p[0], p[1], p[2])
            nw.addData3(n[0], n[1], n[2])
            cw.addData4(c[0], c[1], c[2], c[3])
        prim = GeomTriangles(Geom.UHStatic)
        prim.setIndexType(GeomEnums.NT_uint32)
        for a, b, c in self.T:
            prim.addVertices(a, b, c)
        g = Geom(vd)
        g.addPrimitive(prim)
        gn = GeomNode(name)
        gn.addGeom(g)
        return gn


WHITE = (0.93, 0.93, 0.95, 1)
RED = (0.78, 0.10, 0.10, 1)
DARK = (0.10, 0.15, 0.22, 1)
GREY = (0.32, 0.32, 0.34, 1)
LGREY = (0.62, 0.63, 0.66, 1)
BLACK = (0.05, 0.05, 0.05, 1)


def _add(parent, mesh, name):
    return parent.attachNewNode(mesh.node(name))


def _wheel(parent, x, y, z, r, w=0.1):
    m = Mesh()
    m.loft([(-w, 0, 0, 0), (-w, r, r, 0), (w, r, r, 0), (w, 0, 0, 0)], 14, BLACK)
    m.loft([(-w - 0.02, 0, 0, 0), (-w - 0.02, r * 0.55, r * 0.55, 0), (w + 0.02, r * 0.55, r * 0.55, 0),
            (w + 0.02, 0, 0, 0)], 10, (0.62, 0.63, 0.66, 1))      # cerchione
    n = _add(parent, m, 'ruota')
    n.setPos(x, y, z)
    n.setH(90)


# ---------------------------------------------------------------- modelli
# Modelli 3D dettagliati. Il carrello ha cerniere (g_nose, g_r, g_l, ...) che si ripiegano;
# 'ext' e' l'esterno, 'int' e' la cabina di pilotaggio (visibile solo in vista CABINA).
WIN = (0.07, 0.11, 0.17, 1)
GLASS = (0.12, 0.20, 0.28, 1)
CHROME = (0.78, 0.78, 0.80, 1)
PANEL = (0.10, 0.10, 0.11, 1)
LIGHT_RED = (1.0, 0.1, 0.1, 1)
LIGHT_GREEN = (0.1, 1.0, 0.2, 1)

# ---------------------------------------------------------------- luci notturne (punti luminosi a dimensione fissa)
GLOW = []                 # (NodePath, tipo) delle luci fisse: piste, finestre, lampioni...
_GLOW_FMT = []


def make_points(arr, size, parent=None, name='luci_glow'):
    """arr: Nx7 (x, y, z, r, g, b, a). Punti additivi: restano visibili anche da lontano."""
    arr = np.asarray(arr, float).reshape(-1, 7)
    n = len(arr)
    if not _GLOW_FMT:
        af = GeomVertexArrayFormat()
        af.addColumn(InternalName.getVertex(), 3, Geom.NT_float32, Geom.C_point)
        af.addColumn(InternalName.getColor(), 4, Geom.NT_uint8, Geom.C_color)
        f = GeomVertexFormat()
        f.addArray(af)
        _GLOW_FMT.append(GeomVertexFormat.registerFormat(f))
    vd = GeomVertexData(name, _GLOW_FMT[0], Geom.UH_static)
    vd.uncleanSetNumRows(n)
    rec = np.empty(n, dtype=[('p', '<f4', 3), ('c', 'u1', 4)])
    rec['p'] = arr[:, :3]
    rec['c'] = np.clip(arr[:, 3:7] * 255.0, 0, 255).astype(np.uint8)
    vd.modifyArray(0).modifyHandle().copyDataFrom(rec.tobytes())
    pr = GeomPoints(Geom.UH_static)
    pr.addNextVertices(n)
    g = Geom(vd)
    g.addPrimitive(pr)
    gn = GeomNode(name)
    gn.addGeom(g)
    node = NodePath(gn)
    if parent is not None:
        node.reparentTo(parent)
    node.setLightOff()
    node.setRenderModeThickness(size)
    node.setDepthWrite(False)
    node.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha, ColorBlendAttrib.OOne))
    node.setBin('fixed', 20)
    node.setColorScale(1, 1, 1, 0)
    return node


def glow_pts(pts, col, a=1.0):
    """Lista di (x, y, z) con un colore -> righe Nx7."""
    return [(p[0], p[1], p[2], col[0], col[1], col[2], a) for p in pts]


def add_aircraft_glow(ext, sp):
    """Luci notturne dell'aereo: faro rosso, fari di atterraggio col loro fascio, finestrini illuminati."""
    g = sp.geo
    nw = sp.wheels[0][1]
    make_points([(0.0, nw.y + 0.6, nw.z + 0.9, 1.0, 0.97, 0.85, 1.0), (0.0, nw.y + 0.6, nw.z + 0.9, 1.0, 0.97, 0.85, 1.0)],
                16.0, ext, 'lt_land')
    _cone_beam(ext, 0.0, nw.y + 0.8, nw.z + 0.9, 150.0 if g else 90.0, 18.0 if g else 10.0)
    if g:
        yn, yt, rx, rz, nl, tl = g['fus']
        make_points([(0.0, yn * 0.1, rz + 0.12, 1.0, 0.05, 0.05, 1.0), (0.0, yn * 0.1, -rz - 0.1, 1.0, 0.05, 0.05, 1.0)],
                    10.0, ext, 'lt_beacon')
        rows = []
        y = yn - nl - 3.0
        while y > yt + tl + 3.0:
            for s in (-1, 1):
                rows.append((s * rx * 0.99, y, rz * 0.30, 1.0, 0.82, 0.50, 0.9))
            y -= 1.2
        if rows:
            make_points(rows, 3.0, ext, 'lt_cabin')


def _cone_beam(parent, x, y, z, length=140.0, spread=16.0, drop=0.03):
    """Fascio del faro di atterraggio: cono additivo che parte dal muso e punta in avanti e un po' in basso."""
    m = Mesh()
    seg = 10
    ring = []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        ring.append(m.vert((x + spread * math.cos(a), y + length, z - length * drop + spread * 0.6 * math.sin(a)),
                           (0, 1, 0), (1.0, 0.95, 0.75, 0.0)))
    ap = m.vert((x, y, z), (0, 1, 0), (1.0, 0.95, 0.75, 0.30))
    for i in range(seg):
        m.T.append((ap, ring[i], ring[(i + 1) % seg]))
    n = _add(parent, m, 'lt_beam')
    n.setLightOff()
    n.setTwoSided(True)
    n.setDepthWrite(False)
    n.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha, ColorBlendAttrib.OOne))
    n.setBin('fixed', 21)
    n.setColorScale(1, 1, 1, 0)
    return n


def _beam(m, p0, p1, r, col):
    """Trave a sezione quadrata tra due punti (puntoni, sostegni)."""
    p0, p1 = Vec3(*p0), Vec3(*p1)
    d = p1 - p0
    d.normalize()
    up = Vec3(0, 0, 1) if abs(d.z) < 0.9 else Vec3(1, 0, 0)
    a = d.cross(up)
    a.normalize()
    b = d.cross(a)
    b.normalize()
    pts = []
    for p in (p0, p1):
        for v in (p + a * r + b * r, p + a * r - b * r, p - a * r - b * r, p - a * r + b * r):
            pts.append((v.x, v.y, v.z))
    m.hull(pts, col)


def _lights(parent, pts, r):
    m = Mesh()
    for (x, y, z, col) in pts:
        m.sphere(x, y, z, r, r, r, col, 6, 4)
    _add(parent, m, 'luci').setLightOff()
    # alone luminoso di notte: luci di via (fisse) e strobo bianchi che lampeggiano
    make_points([(x, y, z, c[0], c[1], c[2], 1.0) for (x, y, z, c) in pts], 9.0, parent, 'lt_nav')
    make_points([(x, y, z, 1.0, 1.0, 1.0, 1.0) for (x, y, z, c) in pts], 12.0, parent, 'lt_strobe')


def _dial(parent, x, y, z, r):
    m = Mesh()
    m.loft([(0.0, 0, 0, 0), (0.0, r, r, 0)], 14, LGREY)
    m.loft([(-0.004, 0, 0, 0), (-0.004, r * 0.85, r * 0.85, 0)], 14, BLACK)
    m.box(-0.004, 0.004, -0.008, -0.004, 0.0, r * 0.75, (0.95, 0.95, 0.95, 1))
    n = _add(parent, m, 'quadrante')
    n.setPos(x, y, z)


def _finish_cockpit(it, parts):
    it.setLightOff()
    it.setTwoSided(True)
    it.hide()
    parts['int'] = it


def _cessna_interior(root, parts, sp):
    it = root.attachNewNode('int')
    m = Mesh()
    m.box(-0.62, 0.62, 1.50, 1.74, 0.05, 0.46, PANEL)
    m.box(-0.64, 0.64, 1.15, 1.78, 0.44, 0.50, (0.04, 0.04, 0.05, 1))
    m.box(-0.12, 0.12, 1.20, 1.50, -0.25, 0.10, (0.20, 0.20, 0.22, 1))           # console
    for s in (-1, 1):
        m.box(s * 0.62, s * 0.67, -0.4, 1.45, -0.15, 0.50, (0.55, 0.50, 0.42, 1))  # pannelli porta
        _beam(m, (s * 0.64, 1.50, 0.50), (s * 0.60, 0.85, 1.03), 0.007, (0.12, 0.12, 0.13, 1))  # montanti
    m.box(-0.58, 0.58, 0.80, 0.90, 1.00, 1.06, (0.12, 0.12, 0.13, 1))
    m.box(-0.64, 0.64, -0.5, 0.85, 1.03, 1.08, (0.40, 0.38, 0.34, 1))             # cielo
    _add(it, m, 'cabina')
    for i, x in enumerate((-0.40, -0.24, -0.08, 0.08, 0.24, 0.40)):
        _dial(it, x, 1.499, 0.36, 0.062)
    for x in (-0.36, -0.12, 0.12, 0.36):
        _dial(it, x, 1.499, 0.20, 0.05)
    yk = it.attachNewNode('volante')
    yk.setPos(0, 1.42, 0.30)
    m = Mesh()
    m.box(-0.02, 0.02, -0.05, 0.10, -0.02, 0.02, (0.15, 0.15, 0.16, 1))
    m.box(-0.18, 0.18, -0.38, -0.34, -0.015, 0.015, (0.08, 0.08, 0.09, 1))
    m.box(-0.02, 0.02, -0.38, -0.05, -0.015, 0.015, (0.12, 0.12, 0.13, 1))
    for s in (-1, 1):
        m.box(s * 0.16, s * 0.20, -0.39, -0.33, -0.09, 0.09, (0.06, 0.06, 0.07, 1))
    _add(yk, m, 'yoke')
    parts['ctl'] = [(yk, 'yoke')]
    parts['levers'] = None
    _finish_cockpit(it, parts)


def _airliner_interior(root, parts, sp, yoke):
    it = root.attachNewNode('int')
    ey, ez = sp.eye.y, sp.eye.z
    y0 = ey + 0.95
    m = Mesh()
    m.box(-1.15, 1.15, y0 - 0.3, y0 + 0.5, ez - 0.38, ez - 0.28, (0.05, 0.05, 0.06, 1))
    m.box(-1.15, 1.15, y0, y0 + 0.25, ez - 1.0, ez - 0.30, (0.18, 0.19, 0.21, 1))
    for x in (-0.95, -0.55, -0.2, 0.2, 0.55, 0.95):
        c = (0.02, 0.18, 0.12, 1) if abs(x) < 0.3 else (0.03, 0.12, 0.22, 1)
        m.box(x - 0.15, x + 0.15, y0 - 0.012, y0, ez - 0.78, ez - 0.48, c)
        m.box(x - 0.17, x + 0.17, y0 - 0.006, y0, ez - 0.80, ez - 0.46, (0.02, 0.02, 0.03, 1))
    m.box(-0.18, 0.18, ey - 0.9, y0, ez - 1.25, ez - 0.62, (0.16, 0.16, 0.18, 1))       # piedistallo
    for s in (-1, 1):
        m.box(s * 1.0, s * 1.35, ey - 0.9, y0 + 0.2, ez - 1.0, ez - 0.55, (0.20, 0.21, 0.23, 1))
        m.box(s * 1.05, s * 1.15, y0 + 0.30, y0 + 0.50, ez - 0.30, ez + 0.85, (0.10, 0.10, 0.11, 1))
    m.box(-1.15, 1.15, y0 + 0.30, y0 + 0.60, ez + 0.78, ez + 0.90, (0.10, 0.10, 0.11, 1))
    m.box(-0.9, 0.9, ey - 0.6, ey + 0.2, ez + 0.82, ez + 0.90, (0.45, 0.45, 0.48, 1))   # sopraelevato
    for i in range(8):
        m.box(-0.8 + i * 0.2, -0.65 + i * 0.2, ey - 0.5, ey + 0.1, ez + 0.80, ez + 0.82,
              (0.9, 0.7, 0.1, 1) if i % 3 == 0 else (0.2, 0.7, 0.3, 1))
    _add(it, m, 'cabina')
    lv = it.attachNewNode('manette')
    m = Mesh()
    for dx in (-0.05, 0.05):
        m.box(dx - 0.018, dx + 0.018, 0.0, 0.18, 0.0, 0.14, (0.85, 0.85, 0.87, 1))
    _add(lv, m, 'levers')
    lv.setPos(0, ey - 0.35, ez - 0.62)
    parts['levers'] = lv
    ctl = []
    for s in (-1, 1):
        n = it.attachNewNode('comando')
        mm = Mesh()
        if yoke:
            n.setPos(s * 0.55, y0 - 0.15, ez - 0.55)
            mm.box(-0.02, 0.02, -0.02, 0.3, -0.02, 0.02, (0.15, 0.15, 0.16, 1))
            mm.box(-0.17, 0.17, -0.34, -0.30, -0.015, 0.015, (0.08, 0.08, 0.09, 1))
            for t in (-1, 1):
                mm.box(t * 0.15, t * 0.19, -0.35, -0.29, -0.08, 0.08, (0.06, 0.06, 0.07, 1))
            kind = 'yoke'
        else:
            n.setPos(s * 0.95, y0 - 0.35, ez - 0.50)
            mm.box(-0.025, 0.025, -0.03, 0.03, 0.0, 0.17, (0.12, 0.12, 0.13, 1))
            mm.box(-0.03, 0.03, -0.035, 0.045, 0.14, 0.2, (0.06, 0.06, 0.07, 1))
            kind = 'stick'
        _add(n, mm, 'cmd')
        ctl.append((n, kind))
    parts['ctl'] = ctl
    _finish_cockpit(it, parts)


# ---------------------------------------------------------------- mesh lisce, profili alari e texture (numpy + PIL)
class SMesh:
    """Mesh liscia: griglie di punti (righe x colonne) con normali calcolate, colore per vertice e UV."""

    def __init__(self):
        self.P, self.N, self.C, self.UV, self.T = [], [], [], [], []
        self.n = 0

    def grid(self, P, col=(1, 1, 1, 1), uv=None, closed=True, face=None):
        P = np.asarray(P, float)
        R, S1, _ = P.shape
        if closed:
            Pc = P[:, :S1 - 1]
            du = np.roll(Pc, -1, 1) - np.roll(Pc, 1, 1)
            du = np.concatenate([du, du[:, :1]], 1)
        else:
            du = np.gradient(P, axis=1)
        dv = np.gradient(P, axis=0)
        nrm = np.cross(du, dv)
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-12)
        if face is not None:
            flip = float(np.sum(nrm * np.asarray(face, float))) < 0
        else:
            cen = P[:, :S1 - 1].mean(axis=1, keepdims=True)
            flip = float(np.sum(nrm * (P - cen))) < 0
        if flip:
            nrm = -nrm
        col = np.asarray(col, float)
        if col.ndim == 1:
            col = np.broadcast_to(col, (R, S1, 4))
        if uv is None:
            UV = np.zeros((R, S1, 2))
        elif isinstance(uv, tuple):
            X, Y = np.meshgrid(np.asarray(uv[1], float), np.asarray(uv[0], float))
            UV = np.stack([Y, X], -1)
        else:
            UV = np.asarray(uv, float)
        base = self.n
        self.P.append(P.reshape(-1, 3))
        self.N.append(nrm.reshape(-1, 3))
        self.C.append(col.reshape(-1, 4))
        self.UV.append(UV.reshape(-1, 2))
        idx = base + np.arange(R * S1).reshape(R, S1)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        if flip:
            t = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
        else:
            t = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)])
        self.T.append(t)
        self.n += R * S1

    def node(self, name):
        P = np.concatenate(self.P)
        N = np.concatenate(self.N)
        C = np.concatenate(self.C)
        UV = np.concatenate(self.UV)
        vd = GeomVertexData(name, GeomVertexFormat.getV3n3cpt2(), Geom.UHStatic)
        vd.setNumRows(len(P))
        vw = GeomVertexWriter(vd, 'vertex')
        nw = GeomVertexWriter(vd, 'normal')
        cw = GeomVertexWriter(vd, 'color')
        tw = GeomVertexWriter(vd, 'texcoord')
        for p, n, c, t in zip(P.tolist(), N.tolist(), C.tolist(), UV.tolist()):
            vw.addData3(*p)
            nw.addData3(*n)
            cw.addData4(*c)
            tw.addData2(*t)
        prim = GeomTriangles(Geom.UHStatic)
        prim.setIndexType(GeomEnums.NT_uint32)
        for a, b, c in np.concatenate(self.T).tolist():
            prim.addVertices(a, b, c)
        g = Geom(vd)
        g.addPrimitive(prim)
        gn = GeomNode(name)
        gn.addGeom(g)
        return gn


def _smooth(v, k=2):
    v = np.asarray(v, float)
    pad = np.concatenate([np.repeat(v[:1], k), v, np.repeat(v[-1:], k)])
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), 'valid')


def super_ring(w, a_top, a_bot, zc, y, S=48, nexp=2.3):
    """Sezione trasversale (iper-ellisse) a quota y; angolo 0 = lato destro, parte dal basso."""
    a = np.linspace(-math.pi / 2, 1.5 * math.pi, S + 1)
    c, s = np.cos(a), np.sin(a)
    e = 2.0 / nexp
    x = w * np.sign(c) * np.abs(c) ** e
    z = zc + np.where(s > 0, a_top, a_bot) * np.sign(s) * np.abs(s) ** e
    return np.stack([x, np.full_like(x, y), z], 1)


def _naca(npts, tc, camber=0.0):
    b = np.linspace(0, math.pi, npts + 1)
    x = 0.5 * (1 - np.cos(b))
    yt = 5 * tc * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    yc = camber * 4 * x * (1 - x)
    return x, yt, yc


def foil_mesh(sm, secs, npts=18, cap=True):
    """Superficie alare: ogni sezione e' un dict(le, c, tc, n, b, camber, cu, cd) (bordo d'attacco, corda,
    spessore relativo, normale, direzione indietro, colori sopra/sotto). Profilo NACA a 4 cifre."""
    rings, cols = [], []
    for s in secs:
        x, yt, yc = _naca(npts, s['tc'], s.get('camber', 0.0))
        up = np.stack([x, yc + yt], 1)[::-1]
        dn = np.stack([x, yc - yt], 1)[1:]
        p2 = np.concatenate([up, dn], 0)
        le, b, n = (np.asarray(s[k], float) for k in ('le', 'b', 'n'))
        rings.append(le[None, :] + p2[:, 0:1] * s['c'] * b[None, :] + p2[:, 1:2] * s['c'] * n[None, :])
        cu = np.asarray(s.get('cu', (1, 1, 1, 1)), float)
        cd = np.asarray(s.get('cd', cu), float)
        cols.append(np.concatenate([np.repeat(cu[None], npts + 1, 0), np.repeat(cd[None], npts, 0)]))
    if cap:
        last = rings[-1]
        cen = last.mean(axis=0)
        for sc in (0.7, 0.35, 0.04):
            rings.append(cen + (last - cen) * sc)
            cols.append(cols[-1] if sc != 0.7 else cols[len(secs) - 1])
    sm.grid(np.stack(rings), np.stack(cols))


_TEX_CACHE = {}
TEX_K = [2]          # supersampling delle livree (1 = anteprima veloce nell'hangar)
CUSTOM = {}          # personalizzazioni dell'hangar: nome breve aereo -> dict(body, accent, name, reg, power, weight)
# con PyInstaller (--onefile) __file__ sta in una cartella temporanea: i salvataggi vanno accanto all'.exe
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
HANGAR_FILE = os.path.join(APP_DIR, 'hangar.json')


def _make_tex(im, name):
    t = Texture(name)
    t.setup2dTexture(im.width, im.height, Texture.T_unsigned_byte, Texture.F_rgba8)
    t.setRamImageAs(im.tobytes(), 'RGBA')
    t.setMinfilter(SamplerState.FT_linear_mipmap_linear)
    t.setMagfilter(SamplerState.FT_linear)
    t.setAnisotropicDegree(8)
    t.setWrapU(Texture.WM_clamp)
    t.setWrapV(Texture.WM_repeat)
    return t


def _font(size):
    for f in ('arialbd.ttf', 'Arial Bold.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(f, max(8, int(size)))
        except Exception:
            pass
    return ImageFont.load_default()


class _Painter:
    """Disegna sulla texture di una fusoliera in coordinate (distanza dal muso in m, angolo di sezione)."""

    def __init__(self, L, rx_fn, W, H, K, base):
        self.L, self.W, self.H, self.K = L, W, H, K
        self.pxm = W / L
        self.rx_fn = rx_fn                       # (w, h) semiassi per la scala verticale
        self.im = Image.new('RGB', (W, H), tuple(base))
        self.d = ImageDraw.Draw(self.im, 'RGBA')

    def X(self, dm):
        return dm * self.pxm

    def Y(self, a):
        return (a + math.pi / 2) / (2 * math.pi) * self.H

    def YL(self, a):
        return self.Y(math.pi - a)

    def ppm(self, a):
        """pixel per metro in direzione circonferenziale all'angolo a"""
        w, h = self.rx_fn
        ds = math.sqrt((w * math.sin(a)) ** 2 + (h * math.cos(a)) ** 2)
        return (self.H / (2 * math.pi)) / max(ds, 1e-3)

    def rect(self, dm, a, wm, hm, fill, outline=None, rad=0.06, sides=(0, 1), width=1):
        for sd in sides:
            f = self.Y if sd == 0 else self.YL
            yc = f(a)
            hh = hm * self.ppm(a) / 2
            x0, x1 = self.X(dm - wm / 2), self.X(dm + wm / 2)
            self.d.rounded_rectangle([x0, yc - hh, x1, yc + hh], radius=rad * self.pxm, fill=fill,
                                     outline=outline, width=width * self.K)

    def poly(self, pts, fill, sides=(0, 1), outline=None):
        """pts: lista (distanza m, angolo)"""
        for sd in sides:
            f = self.Y if sd == 0 else self.YL
            self.d.polygon([(self.X(x), f(a)) for x, a in pts], fill=fill, outline=outline)

    def line_x(self, dm, fill, width=1):
        self.d.line([(self.X(dm), 0), (self.X(dm), self.H)], fill=fill, width=width * self.K)

    def line_a(self, a, x0, x1, fill, width=1):
        for f in (self.Y, self.YL):
            self.d.line([(self.X(x0), f(a)), (self.X(x1), f(a))], fill=fill, width=width * self.K)

    def text(self, txt, dm, a, cap_m, color):
        f = _font(cap_m * self.ppm(a) / 0.72)
        if isinstance(txt, (list, tuple)):                       # scritta a piu colori: [(testo, colore), ...]
            full = ''.join(t for t, _ in txt)
            bb = f.getbbox(full)
            lay = Image.new('RGBA', (bb[2] + 6, bb[3] + 6), (0, 0, 0, 0))
            xx, dr = 3, ImageDraw.Draw(lay)
            for t, c in txt:
                dr.text((xx, 0), t, font=f, fill=tuple(c) + (255,))
                xx += f.getlength(t)
        else:
            bb = f.getbbox(txt)
            lay = Image.new('RGBA', (bb[2] + 6, bb[3] + 6), (0, 0, 0, 0))
            ImageDraw.Draw(lay).text((3, 0), txt, font=f, fill=tuple(color) + (255,))
        stretch = self.pxm * 1.0 / (self.ppm(a))
        lay = lay.resize((max(1, int(lay.width * stretch)), lay.height), Image.LANCZOS)
        wm = lay.width / self.pxm
        for sd in (0, 1):
            im2 = lay.rotate(180) if sd == 0 else lay
            yc = (self.Y(a) if sd == 0 else self.YL(a))
            self.im.paste(im2, (int(self.X(dm)), int(yc - im2.height / 2)), im2)
        return wm

    def finish(self, noise=2.0):
        im = self.im.resize((self.W // self.K, self.H // self.K), Image.LANCZOS)
        arr = np.asarray(im).astype(np.int16)
        rng = np.random.default_rng(3)
        arr[..., :3] += rng.normal(0, noise, arr.shape[:2])[..., None].astype(np.int16)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGB').convert('RGBA')


def _col255(c):
    return tuple(int(round(v * 255)) for v in c[:3])


AIRLINES = {'Boeing 737-800': ('SKYWAVE', 'SW'), 'Airbus A320': ('AEROBLU', 'AB'), 'Airbus A380': ('GRAND AIR', 'GA'),
            'Jet privato': ('PRIVATE', 'PJ'), 'FedEx Cargo': ('FedEx', 'FX'), 'DHL Cargo': ('DHL', 'DH'),
            'Amazon Air': ('amazon', 'AZ')}

# livree delle compagnie (colori semplificati, senza loghi): corpo, coda, fascia sotto i finestrini, scritta, timone
LIVERIES = [
    dict(label='Personalizzata (colori scelti)'),
    dict(label='ITA Airways', name='ITA AIRWAYS', body=(0.16, 0.46, 0.92), tail=(0.16, 0.46, 0.92), stripe=(0.16, 0.46, 0.92),
         text=(0.97, 0.97, 0.99), tail2=(0.16, 0.46, 0.92)),
    dict(label='Emirates', name='Emirates', body=(0.97, 0.97, 0.97), tail=(0.97, 0.97, 0.97), stripe=(0.97, 0.97, 0.97),
         text=(0.78, 0.62, 0.30), tail2=(0.97, 0.97, 0.97)),
    dict(label='Air France', name='AIR FRANCE', body=(0.96, 0.96, 0.98), tail=(0.97, 0.97, 0.98), stripe=(0.00, 0.14, 0.45),
         text=(0.00, 0.14, 0.45), tail2=(0.97, 0.97, 0.98)),
    dict(label='British Airways', name='BRITISH AIRWAYS', body=(0.94, 0.94, 0.96), tail=(0.94, 0.94, 0.96),
         stripe=(0.94, 0.94, 0.96), text=(0.10, 0.18, 0.42), tail2=(0.94, 0.94, 0.96)),
    dict(label='Lufthansa', name='LUFTHANSA', body=(0.96, 0.96, 0.98), tail=(0.02, 0.12, 0.40), stripe=(0.02, 0.12, 0.40),
         text=(0.02, 0.12, 0.40), tail2=(1.00, 0.75, 0.00)),
    dict(label='KLM', name='KLM', body=(0.00, 0.63, 0.87), tail=(0.00, 0.63, 0.87), stripe=(0.95, 0.95, 0.97),
         text=(0.97, 0.97, 0.99), tail2=(0.95, 0.95, 0.97)),
    dict(label='Qatar Airways', name='QATAR AIRWAYS', body=(0.96, 0.96, 0.97), tail=(0.35, 0.02, 0.20),
         stripe=(0.35, 0.02, 0.20), text=(0.35, 0.02, 0.20), tail2=(0.55, 0.55, 0.58)),
    dict(label='Singapore Airlines', name='SINGAPORE AIRLINES', body=(0.94, 0.94, 0.96), tail=(0.02, 0.10, 0.34),
         stripe=(0.10, 0.13, 0.26), text=(0.02, 0.10, 0.34), tail2=(0.02, 0.10, 0.34)),
    dict(label='Delta', name='DELTA', body=(0.93, 0.94, 0.97), tail=(0.78, 0.07, 0.17), stripe=(0.00, 0.15, 0.40),
         text=(0.00, 0.15, 0.40), tail2=(0.00, 0.15, 0.40)),
    dict(label='American Airlines', name='AMERICAN', body=(0.78, 0.80, 0.84), tail=(0.05, 0.20, 0.45),
         stripe=(0.75, 0.08, 0.15), text=(0.05, 0.20, 0.45), tail2=(0.75, 0.08, 0.15)),
    dict(label='Ryanair', name='RYANAIR', body=(0.96, 0.96, 0.98), tail=(0.03, 0.22, 0.58), stripe=(0.99, 0.78, 0.00),
         text=(0.03, 0.22, 0.58), tail2=(0.99, 0.78, 0.00)),
    # corrieri: colori ufficiali di marca (FedEx viola #4D148C e arancio #FF6600, DHL giallo #FFCC00 e rosso #D40511, Amazon azzurro Prime)
    dict(label='FedEx Express', name='FedEx', name1='Fed', name2='Ex', body=(0.97, 0.97, 0.98), tail=(0.302, 0.078, 0.549),
         stripe=(0.302, 0.078, 0.549), text=(0.302, 0.078, 0.549), text2=(1.0, 0.40, 0.0), tail2=(1.0, 0.40, 0.0)),
    dict(label='DHL Express', name='DHL', body=(1.0, 0.80, 0.0), tail=(0.831, 0.020, 0.067),
         stripe=(0.831, 0.020, 0.067), text=(0.831, 0.020, 0.067), tail2=(1.0, 0.80, 0.0)),
    dict(label='Amazon Air', name='amazon', body=(0.97, 0.97, 0.98), tail=(0.0, 0.66, 0.88),
         stripe=(0.07, 0.12, 0.18), text=(0.07, 0.12, 0.18), tail2=(1.0, 0.60, 0.0)),
]
LIV_IDX = {l['label'].split()[0]: i for i, l in enumerate(LIVERIES)}      # 'FedEx', 'DHL', 'Amazon'...


def livery_of(cu):
    i = cu.get('livery', 0) if cu else 0
    return LIVERIES[i] if 0 < i < len(LIVERIES) else None


def _tail_decal_image(lv):
    """Emblema sulla deriva: disegno vettoriale semplificato del marchio (non e il logo originale)."""
    S = 512
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    lab = lv.get('label', '')
    W_, Y_, N_ = (250, 250, 252, 255), (255, 200, 0, 255), (10, 30, 100, 255)

    def txt(t, cx, cy, size, col):
        f = _font(size)
        bb = d.textbbox((0, 0), t, font=f)
        d.text((cx - (bb[0] + bb[2]) / 2, cy - (bb[1] + bb[3]) / 2), t, font=f, fill=col)
    def band(o0, o1, col, ang=40.0):                       # fascia inclinata (parallela al bordo d'attacco) tra gli scostamenti o0..o1
        a_ = math.radians(ang)
        dx, dy = math.cos(a_), math.sin(a_)
        nx, ny = -dy, dx
        P = lambda o, tt: (256 + nx * o + dx * tt, 256 + ny * o + dy * tt)
        d.polygon([P(o0, -900), P(o0, 900), P(o1, 900), P(o1, -900)], fill=col)
    if lab.startswith('Lufthansa'):                        # cerchio giallo con gru stilizzata
        d.ellipse([40, 40, 472, 472], fill=Y_)
        d.ellipse([60, 60, 452, 452], outline=N_, width=10)
        d.polygon([(150, 330), (200, 200), (270, 140), (330, 120), (300, 170), (350, 200), (300, 230), (320, 300), (250, 330), (200, 400)], fill=N_)
        d.polygon([(210, 260), (360, 330), (390, 290), (300, 240)], fill=Y_)
    elif lab.startswith('Ryanair'):                        # arpa su disco giallo
        d.ellipse([40, 40, 472, 472], fill=Y_)
        d.arc([150, 110, 370, 400], 200, 340, fill=(3, 56, 148, 255), width=18)
        d.line([(160, 190), (160, 400)], fill=(3, 56, 148, 255), width=16)
        for k in range(6):
            x = 190 + k * 28
            d.line([(x, 170 + k * 8), (x, 380)], fill=(3, 56, 148, 255), width=6)
    elif lab.startswith('ITA'):                              # tricolore diagonale in cima alla deriva: rosso al bordo di uscita, verde davanti
        band(-250, -70, (206, 43, 55, 255), 50.0)
        band(-70, 50, (250, 250, 250, 255), 50.0)
        band(50, 200, (0, 146, 70, 255), 50.0)
    elif lab.startswith('Emirates'):                       # bandiera degli Emirati "aperta" sulla deriva bianca: rosso, verde, nero in diagonale
        band(-80, -15, (214, 20, 36, 255), 52.0)
        band(10, 55, (0, 122, 61, 255), 52.0)
        band(80, 130, (20, 20, 20, 255), 52.0)
    elif lab.startswith('Air France'):                     # bande parallele inclinate blu e rosse su coda bianca
        for i, c in enumerate(((0, 38, 120, 255), (220, 30, 45, 255), (0, 38, 120, 255), (220, 30, 45, 255), (0, 38, 120, 255))):
            o = i * 62
            d.polygon([(0 + o, 512), (50 + o, 512), (330 + o, 0), (280 + o, 0)], fill=c)
    elif lab.startswith('British'):                        # onde rosso / bianco / blu notte / rosso che salgono verso il bordo d'attacco
        base = [(-40, (205, 35, 50, 255), 120), (80, (250, 250, 250, 255), 45), (125, (30, 40, 85, 255), 105), (230, (205, 35, 50, 255), 120), (350, (250, 250, 250, 255), 60)]
        for off, c, th in base:
            top = [(x, off + 150 - 0.55 * x + 28 * math.sin(x * 0.014)) for x in range(0, 520, 20)]
            d.polygon(top + [(p[0], p[1] + th) for p in top[::-1]], fill=c)
    elif lab.startswith('KLM'):                            # corona stilizzata
        d.polygon([(110, 380), (90, 190), (180, 260), (256, 130), (332, 260), (422, 190), (402, 380)], fill=W_)
        d.rectangle([110, 390, 402, 430], fill=W_)
        txt('KLM', 256, 470, 70, W_)
    elif lab.startswith('Qatar'):
        d.ellipse([60, 60, 452, 452], fill=(185, 190, 196, 255))
        d.polygon([(150, 370), (190, 230), (290, 140), (370, 150), (330, 200), (360, 250), (300, 290), (280, 380), (230, 420)], fill=(92, 8, 50, 255))
    elif lab.startswith('Singapore'):                      # coda blu notte: uccello dorato (tre ali a chevron verso l'alto) e filo dorato sul bordo d'attacco
        gold = (238, 176, 30, 255)
        band(-135, -118, gold, 52.0)
        for k in range(3):
            o = k * 58
            d.polygon([(110 + o, 300 - o * 0.35), (300 + o * 0.3, 150 - o * 0.3), (350 + o * 0.3, 205 - o * 0.3), (200 + o, 330 - o * 0.35)], fill=gold)
        d.polygon([(300, 160), (430, 70), (400, 190), (340, 230)], fill=gold)
    elif lab.startswith('Delta'):                          # widget a triangolo
        d.polygon([(256, 60), (470, 440), (42, 440)], fill=(200, 20, 50, 255))
        d.polygon([(256, 170), (400, 440), (112, 440)], fill=(0, 40, 105, 255))
    elif lab.startswith('American'):
        for k, c in enumerate(((185, 190, 198, 255), (200, 20, 40, 255), (10, 50, 120, 255))):
            d.polygon([(60 + k * 120, 470), (150 + k * 120, 470), (310 + k * 120, 50), (220 + k * 120, 50)], fill=c)
    elif lab.startswith('FedEx'):
        txt('Fed', 256, 190, 170, (77, 20, 140, 255))
        txt('Ex', 256, 340, 170, (255, 102, 0, 255))
    elif lab.startswith('DHL'):
        d.rectangle([40, 170, 472, 340], fill=Y_)
        txt('DHL', 256, 256, 150, (212, 5, 17, 255))
    elif lab.startswith('Amazon'):
        txt('prime', 256, 200, 130, W_)
        d.arc([90, 220, 420, 420], 20, 160, fill=(255, 153, 0, 255), width=26)
        d.polygon([(405, 300), (440, 250), (455, 330)], fill=(255, 153, 0, 255))
    else:
        return None
    return im


def tail_decal_texture(lv):
    key = 'decal_' + lv.get('label', '?')
    if key in _TEX_CACHE:
        return _TEX_CACHE[key]
    im = _tail_decal_image(lv)
    if im is None:
        return None
    t = _make_tex(im, key)
    _TEX_CACHE[key] = t
    return t


def uv_poly(pts, uvs, tex, name='emblema'):
    """Poligono convesso con coordinate texture (ventaglio di triangoli)."""
    vd = GeomVertexData(name, GeomVertexFormat.getV3t2(), Geom.UHStatic)
    vd.setNumRows(len(pts))
    wv, wt = GeomVertexWriter(vd, 'vertex'), GeomVertexWriter(vd, 'texcoord')
    for p_, u_ in zip(pts, uvs):
        wv.addData3(*p_)
        wt.addData2(*u_)
    pr = GeomTriangles(Geom.UHStatic)
    for i in range(1, len(pts) - 1):
        pr.addVertices(0, i, i + 1)
    g = Geom(vd)
    g.addPrimitive(pr)
    gn = GeomNode(name)
    gn.addGeom(g)
    n = NodePath(gn)
    n.setTexture(tex)
    n.setTransparency(TransparencyAttrib.MAlpha)
    n.setTwoSided(True)
    return n


def airliner_texture(sp):
    key = sp.short
    if key in _TEX_CACHE:
        return _TEX_CACHE[key]
    g = sp.geo
    yn, yt, rx, rz, nl, tl = g['fus']
    L = yn - yt
    big = rx > 3.0
    K = TEX_K[0]
    cu = CUSTOM.get(key, {})
    body = _col255(g['body'])
    acc = _col255(g['accent'])
    lv = livery_of(cu)
    stripe_c = _col255(lv['stripe']) if lv else acc
    text_c = _col255(lv['text']) if lv else acc
    tail2_c = _col255(lv['tail2']) if lv else None
    P = _Painter(L, (rx, rz), 4096 * K, 1024 * K, K, body)
    d = P.d
    glass = (20, 30, 44, 255)
    frame = (205, 208, 214, 255)
    # pancia grigia, radome, fine coda
    belly = (194, 198, 206, 255)
    d.rectangle([0, 0, P.W, P.Y(-0.62)], fill=belly)
    d.rectangle([0, P.Y(math.pi + 0.62), P.W, P.H], fill=belly)
    d.rectangle([0, 0, P.X(nl + 1.0), P.H], fill=body + (255,))      # muso tutto bianco
    d.rectangle([P.X(L - tl * 0.06), 0, P.W, P.H], fill=(96, 98, 104, 255))
    # coda colorata (sul dorso, sale verso la deriva) e fascia sotto i finestrini
    P.poly([(L - 17.0 * rz / 2, 1.5), (L - 8.0, 0.78), (L, 0.15), (L, math.pi / 2), (L - 17.0 * rz / 2, math.pi / 2)], acc + (255,))
    if tail2_c:          # seconda tinta sulla coda (tricolore Air France, filetti BA, ecc.)
        P.poly([(L - 12.0 * rz / 2, 1.5), (L - 6.2, 0.98), (L - 1.0, 0.45), (L - 1.0, 0.25), (L - 6.2, 0.74),
                (L - 12.0 * rz / 2 - 1.2, 1.5)], tail2_c + (255,))
    x0 = nl + 3.0
    n_ = 24
    top, bot = [], []
    for i in range(n_ + 1):
        t = i / n_
        x = x0 + (L - 1.2 - x0) * t
        ab = -0.30 + 0.62 * t ** 3.0
        th = 0.07 + 0.20 * t
        bot.append((x, ab))
        top.append((x, ab + th))
    P.poly(top + bot[::-1], stripe_c + (255,))
    # linee dei pannelli
    for x in np.arange(2.0, L - 1.0, 2.6 if not big else 3.4):
        P.line_x(x, (110, 114, 124, 60))
    for a in (0.95, -0.72):
        P.line_a(a, 2.0, L - 2.0, (110, 114, 124, 55))
    # finestrini
    pitch = 0.508 if not big else 0.62
    wm, hm = (0.24, 0.34) if not big else (0.28, 0.40)
    doors = [nl + 2.2, L - tl - 6.5] if not big else [nl + 2.4, nl + 20.0, L - tl - 9.0]
    rows = [(0.5, nl + 1.2, L - tl - 2.0)] if not big else [(-0.25, nl + 3.0, L - tl - 3.0), (1.75, nl + 3.0, L - tl - 14.0)]
    cargo = getattr(sp, 'cargo', False)
    if cargo:
        rows = []                                                      # i cargo non hanno finestrini
    for z, xa, xb in rows:
        a = math.asin(max(-0.9, min(0.9, z / rz)))
        x = xa
        while x < xb:
            if all(abs(x - dd) > 1.05 for dd in doors) or z > 1.0:
                P.rect(x, a, wm, hm, glass, outline=frame, rad=0.07, width=1)
            x += pitch
    # porte con oblo'
    for dd in doors:
        zc = -0.2 if not big else -0.8
        a = math.asin(zc / rz)
        P.rect(dd, a, 0.95 if not big else 1.1, 1.9 if not big else 2.3, (0, 0, 0, 0), outline=(120, 124, 134, 255),
               rad=0.18, width=2)
        P.rect(dd, math.asin((zc + (0.7 if not big else 0.9)) / rz), 0.25, 0.35, glass, outline=frame, rad=0.07)
    if cargo:                                                          # grande portellone merci sul fianco
        dk = tuple(int(c * 0.82) for c in body) + (255,)
        P.rect(nl + 9.0, math.asin(0.15 / rz), 3.6, 2.6, dk, outline=(110, 114, 124, 255), rad=0.12, width=2)
        P.rect(nl + 9.0, math.asin(0.15 / rz), 3.2, 2.2, (0, 0, 0, 0), outline=(150, 154, 164, 255), rad=0.1, width=1)
        P.line_x(nl + 9.0, (90, 94, 104, 140))
    # cabina di pilotaggio sul muso
    d0, wl = nl * 0.56, nl * 0.26
    for sd in (0, 1):
        f = P.Y if sd == 0 else P.YL
        pane = [(d0, 1.50), (d0, 0.98), (d0 + wl, 0.90), (d0 + wl * 0.95, 1.52)]
        P.poly(pane, glass, sides=(sd,), outline=frame)
        P.poly([(d0 + wl * 1.05, 0.88), (d0 + wl * 1.9, 0.62), (d0 + wl * 1.9, 1.0), (d0 + wl * 1.05, 1.1)],
               glass, sides=(sd,), outline=frame)
        P.poly([(d0 + wl * 1.95, 0.60), (d0 + wl * 2.6, 0.55), (d0 + wl * 2.6, 0.9), (d0 + wl * 1.95, 0.95)],
               glass, sides=(sd,), outline=frame)
    # scritte
    name, reg = AIRLINES.get(key, ('SKYWAVE', 'SW'))
    name = cu.get('name') or (lv['name'] if lv else name)
    if getattr(sp, 'cargo', False) and lv:
        name = lv['name']
    regtxt = cu.get('reg') or 'N%s%d' % (reg, 100 + len(key))
    capm = 0.72 if not big else 1.25
    zt = 1.25 if not big else 3.0
    a_t = math.asin(min(0.9, zt / rz))
    if cargo:
        capm = 1.05
    if lv and lv.get('name1'):                                         # scritta a due colori (FedEx)
        P.text([(lv['name1'], text_c), (lv['name2'], _col255(lv['text2']))], nl + 14.0, a_t, capm, text_c)
    else:
        wtxt = P.text(name, nl + (14.0 if cargo else 4.0), a_t, capm, text_c)
    P.text(regtxt, L - tl - (2.0 if cargo else 11.0),math.asin(min(0.9, (zt - 0.2) / rz)), capm * 0.45, (60, 62, 70))
    im = P.finish()
    tex = _make_tex(im, 'livrea_' + key)
    _TEX_CACHE[key] = tex
    return tex


def cessna_texture():
    if 'cessna' in _TEX_CACHE:
        return _TEX_CACHE['cessna']
    y0, y1 = 2.52, -5.9
    L = y0 - y1
    K = TEX_K[0]
    cu = CUSTOM.get('Cessna 172', {})
    P = _Painter(L, (0.64, 0.83), 2048 * K, 1024 * K, K, _col255(cu.get('body', (0.93, 0.93, 0.96))))
    d = P.d
    red = _col255(cu.get('accent', (0.74, 0.11, 0.13))) + (255,)
    blue = (26, 52, 110, 255)
    glass = (34, 48, 62, 255)
    d.rectangle([0, 0, P.X(0.12), P.H], fill=(40, 40, 44, 255))
    # fasce rosso/blu dal cofano alla coda
    def band(a0, a1, col):
        top, bot = [], []
        for i in range(25):
            t = i / 24
            x = 0.5 + (L - 0.6) * t
            lift = 0.0 if t < 0.55 else 0.65 * ((t - 0.55) / 0.45) ** 2
            top.append((x, a1 + lift))
            bot.append((x, a0 + lift * 0.7))
        P.poly(top + bot[::-1], col)
    band(-0.12, 0.10, red)
    band(-0.30, -0.19, blue)
    # vetri: parabrezza, portiere, finestrino posteriore, lunotto
    def u(y):
        return y1 and (y0 - y)
    P.poly([(u(1.40), 1.25), (u(1.40), 1.90), (u(0.66), 1.95), (u(0.66), 1.20)], glass, sides=(0,), outline=(235, 235, 238, 255))
    for (xa, xb, aa, ab) in ((1.12, 0.14, 0.42, 1.02), (-0.10, -1.28, 0.44, 0.98)):
        P.poly([(u(xa), aa), (u(xb), aa), (u(xb - 0.02), ab - 0.06), (u(xa - 0.38), ab)], glass)
    P.poly([(u(-1.38), 0.95), (u(-2.15), 1.20), (u(-2.15), math.pi - 1.20), (u(-1.38), math.pi - 0.95)], glass, sides=(0,))
    P.poly([(u(0.62), math.pi / 2 - 0.55), (u(0.62), math.pi / 2 + 0.55), (u(1.40), math.pi / 2 + 0.7), (u(1.40), math.pi / 2 - 0.7)],
           glass, sides=(0,))
    # maniglia portiera + scritta di immatricolazione
    P.rect(u(0.55), 0.35, 0.14, 0.04, (150, 150, 156, 255), rad=0.02)
    P.text(cu.get('reg') or 'N172SK', u(-3.4), 0.30, 0.30, (60, 60, 68))
    for xx in (1.4, 0.0, -1.5, -3.0, -4.4):
        P.line_x(u(xx), (110, 114, 124, 50))
    im = P.finish()
    tex = _make_tex(im, 'cessna_livrea')
    _TEX_CACHE['cessna'] = tex
    return tex


def fan_texture():
    if 'fan' in _TEX_CACHE:
        return _TEX_CACHE['fan']
    N = 256
    im = Image.new('RGBA', (N, N), (28, 28, 32, 255))
    d = ImageDraw.Draw(im)
    c = N / 2
    nb = 24
    for i in range(nb):
        a0 = 2 * math.pi * i / nb
        a1 = a0 + 2 * math.pi / nb * 0.75
        pts = [(c + 0.2 * c * math.cos(a0), c + 0.2 * c * math.sin(a0)),
               (c + 0.98 * c * math.cos(a0 + 0.25), c + 0.98 * c * math.sin(a0 + 0.25)),
               (c + 0.98 * c * math.cos(a1 + 0.25), c + 0.98 * c * math.sin(a1 + 0.25)),
               (c + 0.2 * c * math.cos(a1), c + 0.2 * c * math.sin(a1))]
        d.polygon(pts, fill=(150, 152, 160, 255) if i % 2 else (118, 120, 128, 255))
    d.ellipse([c - 0.97 * c, c - 0.97 * c, c + 0.97 * c, c + 0.97 * c], outline=(60, 62, 68, 255), width=3)
    tex = _make_tex(im.resize((128, 128), Image.LANCZOS), 'fan')
    _TEX_CACHE['fan'] = tex
    return tex


def _ctrl(parent, name, pos, pts_bot, pts_top, col):
    """Superficie mobile: cuneo (hull) con cerniera in pos; i punti sono assoluti."""
    n = parent.attachNewNode(name)
    n.setPos(*pos)
    m = Mesh()
    rel = lambda p: (p[0] - pos[0], p[1] - pos[1], p[2] - pos[2])
    m.hull([rel(p) for p in pts_bot] + [rel(p) for p in pts_top], col)
    _add(n, m, name + '_g')
    return n


def _engine(ext, ex, ey, ez, er, el, eng, acc, fan):
    sm = SMesh()
    S = 36
    ang = np.linspace(0, 2 * math.pi, S + 1)

    def ring(y, r):
        return np.stack([r * np.cos(ang), np.full_like(ang, y), r * np.sin(ang)], 1)

    prof = [(0.50, 0.95), (0.485, 0.99), (0.45, 1.03), (0.25, 1.07), (-0.05, 1.02), (-0.30, 0.92), (-0.47, 0.84), (-0.50, 0.80)]
    rows = [ring(f * el, r * er) for f, r in prof]
    cols = [(acc if i < 3 else eng) for i in range(len(prof))]
    sm.grid(np.stack(rows), np.stack([np.broadcast_to(np.asarray(c, float), (S + 1, 4)) for c in cols]))
    inner = [(0.502, 0.95), (0.50, 0.90), (0.47, 0.87), (0.40, 0.86), (0.36, 0.86)]
    dark = np.asarray((0.16, 0.16, 0.18, 1.0))
    sm.grid(np.stack([ring(f * el, r * er) for f, r in inner]), dark)
    core = [(-0.50, 0.80), (-0.505, 0.58), (-0.54, 0.56), (-0.5 - 0.5 * er / el, 0.46), (-0.5 - 1.3 * er / el, 0.12)]
    sm.grid(np.stack([ring(f * el, r * er) for f, r in core]), np.asarray((0.42, 0.42, 0.45, 1.0)))
    # ogiva
    yb = 0.36 * el
    spin = [(yb, 0.20), (yb + 0.12 * er, 0.19), (yb + 0.32 * er, 0.13), (yb + 0.55 * er, 0.05), (yb + 0.62 * er, 0.004)]
    sm.grid(np.stack([ring(y, r * er) for y, r in spin]), np.asarray((0.93, 0.93, 0.95, 1.0)))
    n = _add(ext, sm, 'motore')
    n.setPos(ex, ey, ez)
    # disco della ventola con texture
    fm = SMesh()
    rr = np.linspace(0.02, 0.86 * er, 6)
    P = np.stack([np.stack([r * np.cos(ang), np.full_like(ang, yb), r * np.sin(ang)], 1) for r in rr])
    uv = np.stack([0.5 + 0.5 * P[..., 0] / (0.86 * er), 0.5 + 0.5 * P[..., 2] / (0.86 * er)], -1)
    fm.grid(P, (1, 1, 1, 1), uv=uv, face=(0, 1, 0))
    fn = _add(ext, fm, 'ventola')
    fn.setPos(ex, ey, ez)
    fn.setTexture(fan_texture())


def build_cessna_model(sp):
    parts = {}
    root = NodePath('aereo')
    ext = root.attachNewNode('ext')
    parts['ext'] = ext
    ys = np.array([2.52, 2.50, 2.35, 2.10, 1.80, 1.45, 1.10, 0.55, -0.20, -1.00, -1.60, -2.40, -3.40, -4.50, -5.30, -5.90])
    top = np.array([0.04, 0.40, 0.44, 0.47, 0.50, 0.58, 0.90, 1.07, 1.08, 1.04, 0.90, 0.62, 0.50, 0.40, 0.36, 0.32])
    bot = np.array([-0.04, -0.33, -0.40, -0.47, -0.55, -0.62, -0.64, -0.64, -0.62, -0.52, -0.42, -0.25, -0.08, 0.08, 0.14, 0.22])
    wid = np.array([0.02, 0.44, 0.52, 0.58, 0.61, 0.63, 0.64, 0.64, 0.64, 0.60, 0.52, 0.40, 0.28, 0.17, 0.09, 0.04])
    yy = np.concatenate([np.linspace(2.52, 2.40, 4), np.linspace(2.35, -5.9, 56)])
    ti = _smooth(np.interp(yy, ys[::-1], top[::-1]), 1)
    bi = _smooth(np.interp(yy, ys[::-1], bot[::-1]), 1)
    wi = _smooth(np.interp(yy, ys[::-1], wid[::-1]), 1)
    rows = [super_ring(max(w, 0.004), max((t - b) / 2, 0.004), max((t - b) / 2, 0.004), (t + b) / 2, y, 40, 2.5)
            for y, t, b, w in zip(yy, ti, bi, wi)]
    sm = SMesh()
    sm.grid(np.stack(rows), (1, 1, 1, 1), uv=((2.52 - yy) / 8.42, np.linspace(0, 1, 41)))
    n = _add(ext, sm, 'fusoliera')
    n.setTexture(cessna_texture())

    cu_ = CUSTOM.get('Cessna 172', {})
    RED_ = tuple(cu_.get('accent', (0.74, 0.10, 0.12))[:3]) + (1,)
    WH = tuple(min(1.0, v + 0.02) for v in cu_.get('body', (0.93, 0.93, 0.95))[:3]) + (1,)
    BK = (1, -0.0, 0.0, 0)
    # ala alta: profilo NACA 2412, rastremata, con punta rossa
    def wing_secs(s):
        out = []
        for f in (0.0, 0.3, 0.6, 0.9, 1.0):
            x = 0.55 + (5.5 - 0.55) * f
            le = 0.9 + (0.5 - 0.9) * f
            te = -0.85 + (-0.62 + 0.85) * f
            c = le - te
            out.append(dict(le=(s * x, le, 1.10 + 0.07 * f + 0.06 * c / 1.7), c=c, tc=0.12, camber=0.02,
                            n=(0, 0, 1), b=(0, -1, 0), cu=RED_ if f >= 0.995 else WH, cd=(0.80, 0.80, 0.84, 1)))
        return out

    sw = SMesh()
    for s in (-1, 1):
        foil_mesh(sw, wing_secs(s))
    _add(ext, sw, 'ala')
    m = Mesh()
    for s in (-1, 1):
        _beam(m, (s * 0.5, 0.15, -0.5), (s * 3.2, 0.15, 1.08), 0.04, WH)
        _beam(m, (s * 1.9, 0.15, 0.2), (s * 2.1, 0.0, 1.08), 0.025, WH)
    _beam(m, (-2.5, 0.9, 1.1), (-2.5, 1.35, 1.1), 0.012, (0.2, 0.2, 0.2, 1))
    _add(ext, m, 'montanti')
    _lights(ext, ((-5.57, 0.0, 1.17, LIGHT_RED), (5.57, 0.0, 1.17, LIGHT_GREEN)), 0.07)

    # impennaggi
    st = SMesh()
    for s in (-1, 1):
        foil_mesh(st, [dict(le=(s * 0.25, -4.45, 0.30), c=1.0, tc=0.09, n=(0, 0, 1), b=(0, -1, 0), cu=WH),
                       dict(le=(s * 1.0, -4.62, 0.31), c=0.85, tc=0.08, n=(0, 0, 1), b=(0, -1, 0), cu=WH),
                       dict(le=(s * 1.8, -4.82, 0.32), c=0.6, tc=0.07, n=(0, 0, 1), b=(0, -1, 0), cu=WH)])
    foil_mesh(st, [dict(le=(0, -3.55, 0.30), c=1.9, tc=0.09, n=(1, 0, 0), b=(0, -1, 0), cu=RED_),
                   dict(le=(0, -4.2, 1.0), c=1.45, tc=0.08, n=(1, 0, 0), b=(0, -1, 0), cu=RED_),
                   dict(le=(0, -4.85, 1.78), c=0.85, tc=0.07, n=(1, 0, 0), b=(0, -1, 0), cu=RED_)])
    _add(ext, st, 'coda')
    pos = (0, -5.22, 0.31)
    parts['elev'] = _ctrl(ext, 'elev', pos,
                          [(-1.7, -5.24, 0.29), (1.7, -5.24, 0.29), (1.7, -5.43, 0.305), (-1.7, -5.43, 0.305)],
                          [(-1.7, -5.24, 0.34), (1.7, -5.24, 0.34), (1.7, -5.43, 0.315), (-1.7, -5.43, 0.315)], WH)
    pos = (0, -5.15, 1.0)
    parts['rud'] = _ctrl(ext, 'rud', pos,
                         [(-0.045, -4.88, 0.36), (0.045, -4.88, 0.36), (0.02, -5.47, 0.36), (-0.02, -5.47, 0.36)],
                         [(-0.04, -5.40, 1.74), (0.04, -5.40, 1.74), (0.02, -5.69, 1.74), (-0.02, -5.69, 1.74)], RED_)
    for s, key in ((1, 'ail_r'), (-1, 'ail_l')):
        x0, x1 = (2.9, 5.15) if s > 0 else (-5.15, -2.9)
        parts[key] = _ctrl(ext, key, (0, -0.40, 1.195),
                           [(x0, -0.42, 1.149), (x1, -0.38, 1.173), (x0, -0.74, 1.179), (x1, -0.645, 1.201)][0:1] +
                           [(x1, -0.38, 1.173), (x1, -0.645, 1.201), (x0, -0.74, 1.179)],
                           [(x0, -0.42, 1.224), (x1, -0.38, 1.244), (x1, -0.645, 1.211), (x0, -0.74, 1.189)], WH)

    gear = ext.attachNewNode('carrello')
    m = Mesh()
    for sx in (-1, 1):
        m.box(min(sx * 0.2, sx * 1.3), max(sx * 0.2, sx * 1.3), -0.45, -0.25, -0.72, -0.62, GREY)
    _add(ext, m, 'monconi')
    nn = gear.attachNewNode('g_nose')
    nn.setPos(0, 1.9, -0.5)
    m = Mesh()
    m.box(-0.05, 0.05, -0.06, 0.06, -0.35, 0, GREY)
    m.box(-0.03, 0.03, -0.04, 0.04, -0.7, -0.35, CHROME)
    m.box(-0.03, 0.03, 0.04, 0.12, -0.6, -0.4, GREY)
    _add(nn, m, 'gamba')
    _wheel(nn, 0, 0, -0.7, 0.3, 0.07)
    parts['g_nose'] = nn
    for s, key in ((1, 'g_r'), (-1, 'g_l')):
        n = gear.attachNewNode(key)
        n.setPos(s * 1.3, -0.35, -0.65)
        m = Mesh()
        m.box(-0.05, 0.05, -0.05, 0.05, -0.55, 0, GREY)
        m.box(-0.03, 0.03, -0.03, 0.03, -0.55, -0.3, CHROME)
        _add(n, m, 'gamba')
        _wheel(n, 0, 0, -0.55, 0.3, 0.08)
        parts[key] = n
    parts['gear'] = gear

    prop = root.attachNewNode('elica')                  # resta visibile anche in cabina
    prop.setPos(0, 2.52, -0.0)
    sp_ = SMesh()
    ang = np.linspace(0, 2 * math.pi, 25)
    stn = [(0.0, 0.225), (0.08, 0.215), (0.22, 0.175), (0.42, 0.10), (0.58, 0.035), (0.64, 0.004)]
    sp_.grid(np.stack([np.stack([r * np.cos(ang), np.full_like(ang, y), r * np.sin(ang)], 1) for y, r in stn]),
             np.asarray(RED_))
    for sgn in (-1, 1):                                  # pale a corda variabile
        z0, z1 = 0.15, 1.05
        secs = []
        for k, f in enumerate((0.0, 0.5, 1.0)):
            z = sgn * (z0 + (z1 - z0) * f)
            secs.append(dict(le=(0.0, 0.05 + 0.02 * f, z), c=0.18 - 0.07 * f, tc=0.1, n=(1, 0, 0), b=(0, -1, 0),
                             cu=(0.07, 0.07, 0.08, 1) if f < 0.9 else (0.95, 0.8, 0.1, 1)))
        foil_mesh(sp_, secs, 8)
    _add(prop, sp_, 'pale')
    parts['prop'] = prop

    _cessna_interior(root, parts, sp)
    root.setTwoSided(True)
    return root, parts


def build_airliner_model(sp):
    g = sp.geo
    parts = {}
    root = NodePath('aereo')
    ext = root.attachNewNode('ext')
    parts['ext'] = ext
    body, acc, eng = g['body'], g['accent'], g['engine']
    yn, yt, rx, rz, nl, tl = g['fus']
    L = yn - yt
    big = rx > 3.0
    S = 48

    # ---- fusoliera: muso ogivale che scende, cilindro, cono di coda rialzato
    rows = []
    for t in np.linspace(0, 1, 18) ** 1.5:
        t = max(t, 0.004)
        f = max(t ** 0.62, 0.03)
        # la linea inferiore resta piatta; scende solo quella superiore (parabrezza -> punta)
        bot = -rz + 0.16 * rz * (1 - t) ** 4
        hh = rz * f
        rows.append((yn - nl * t, rx * f, hh, hh, bot + hh))
    rows.append((yn - nl - (L - nl - tl) * 0.5, rx, rz, rz, 0.0))
    rows.append((yt + tl, rx, rz, rz, 0.0))
    for t in np.linspace(0, 1, 14)[1:]:
        f = 0.07 + 0.93 * (1 - t ** 1.6) ** 0.8
        rows.append((yt + tl * (1 - t), rx * f, rz * f * 0.98, rz * f, rz * 0.55 * t ** 1.6))
    ys_ = np.array([r[0] for r in rows])
    grid = np.stack([super_ring(w, at, ab, zc, y, S, 2.25) for (y, w, at, ab, zc) in rows])
    sm = SMesh()
    sm.grid(grid, (1, 1, 1, 1), uv=((yn - ys_) / L, np.linspace(0, 1, S + 1)))
    n = _add(ext, sm, 'fusoliera')
    n.setTexture(airliner_texture(sp))

    # ---- ala (NACA, rastremata, con freccia e winglet) e carena ala-fusoliera
    rxw, rle, rte, txw, tle, tte, wz, th, dz = g['wing']

    def wing_at(x):
        u = (x - rxw) / (txw - rxw)
        return rle + (tle - rle) * u, rte + (tte - rte) * u, wz + dz * u, 0.13 - 0.04 * u

    style = sp.short
    if big:
        H, phimax, swp, ratio = 1.9, 86.0, 0.6, 0.55
    elif g.get('rear'):
        H, phimax, swp, ratio = 1.25, 72.0, 0.9, 0.6
    elif 'A320' in style:
        H, phimax, swp, ratio = 2.4, 74.0, 1.4, 0.55
    else:
        H, phimax, swp, ratio = 2.2, 80.0, 1.8, 0.62
    # percorso curvo della winglet: la direzione ruota da orizzontale a quasi verticale
    taus = np.linspace(0, 1, 7)[1:]
    phis = np.radians(phimax) * taus ** 1.3
    seg = np.stack([np.cos(phis), np.sin(phis)], 1)
    cum = np.cumsum(seg, axis=0)
    cum *= H / cum[-1, 1]
    lean = float(cum[-1, 0])
    BODY = np.asarray(body)
    GRY = np.asarray((0.78, 0.79, 0.83, 1.0))
    ACC = np.asarray(acc)
    sw = SMesh()
    for s in (-1, 1):
        secs = []
        for f in (0.0, 0.18, 0.4, 0.65, 0.85, 1.0):
            x = rxw + (txw - rxw) * f
            le, te, zb, tc = wing_at(x)
            c = le - te
            secs.append(dict(le=(s * x, le, zb + 0.5 * tc * c), c=c, tc=tc, camber=0.015, n=(0, 0, 1), b=(0, -1, 0),
                             cu=BODY, cd=GRY))
        le_t, te_t, zb_t, tc_t = wing_at(txw)
        c_t = le_t - te_t
        z_t = zb_t + 0.5 * tc_t * c_t
        for k, tau in enumerate(taus):
            xo = txw + cum[k, 0] * 0.9
            zo = z_t + cum[k, 1]
            leo = le_t - swp * tau
            co = c_t * (1.35 - (1.35 - ratio) * tau)
            dpath = np.array([s * math.cos(phis[k]), 0.0, math.sin(phis[k])])
            nvec = np.cross(np.array([0, -1.0, 0]), dpath)
            nvec /= np.linalg.norm(nvec)
            secs.append(dict(le=(s * xo, leo, zo), c=co, tc=0.10, n=tuple(nvec), b=(0, -1, 0),
                             cu=ACC if (not big and tau > 0.55) else BODY, cd=GRY))
        foil_mesh(sw, secs, 18)
    # carena ala-fusoliera
    fr = [(rle + 1.0, 0.02, 0.02, -rz * 0.80), (rle - 0.5, rx * 0.88, rz * 0.20, -rz * 0.86),
          (rte + 0.5, rx * 0.88, rz * 0.20, -rz * 0.86), (rte - 4.0, rx * 0.62, rz * 0.14, -rz * 0.86),
          (rte - 6.5, 0.02, 0.02, -rz * 0.82)]
    yy = np.linspace(fr[0][0], fr[-1][0], 14)
    fyv = np.array([r[0] for r in fr])[::-1]
    fw = _smooth(np.interp(yy[::-1], fyv, np.array([r[1] for r in fr])[::-1])[::-1], 1)
    fh = _smooth(np.interp(yy[::-1], fyv, np.array([r[2] for r in fr])[::-1])[::-1], 1)
    fz = _smooth(np.interp(yy[::-1], fyv, np.array([r[3] for r in fr])[::-1])[::-1], 1)
    sw.grid(np.stack([super_ring(max(w, 0.02), max(h, 0.02), max(h, 0.02), z, y, 24, 2.0)
                      for y, w, h, z in zip(yy, fw, fh, fz)]), np.asarray((0.80, 0.81, 0.84, 1.0)))
    _add(ext, sw, 'ali')

    # flap, spoiler e alettoni come pannelli (cuneo appena sporgente)
    mp = Mesh()
    n_sp = 6 if big else 5
    for s in (-1, 1):
        for i in range(n_sp):
            xa = rxw + (txw - rxw) * (0.16 + 0.56 * i / n_sp)
            xb = xa + (txw - rxw) * 0.56 / n_sp * 0.92
            la, ta, za, tca = wing_at(xa)
            lb, tb, zb, tcb = wing_at(xb)
            ca, cb = la - ta, lb - tb
            ha, hb = za + tca * ca * 0.62, zb + tcb * cb * 0.62
            d_ = 0.03
            mp.hull([(s * xa, ta + 0.34 * ca, ha - 0.0), (s * xa, ta + 0.10 * ca, ha - 0.03), (s * xb, tb + 0.10 * cb, hb - 0.03),
                     (s * xb, tb + 0.34 * cb, hb - 0.0), (s * xa, ta + 0.34 * ca, ha + d_), (s * xa, ta + 0.10 * ca, ha + d_ * 0.6),
                     (s * xb, tb + 0.10 * cb, hb + d_ * 0.6), (s * xb, tb + 0.34 * cb, hb + d_)], (0.70, 0.72, 0.75, 1))
        for (xa, xb) in ((rxw + 0.5, (rxw + txw) * 0.42), ((rxw + txw) * 0.45, txw * 0.58)):
            la, ta, za, tca = wing_at(xa)
            lb, tb, zb, tcb = wing_at(xb)
            mp.hull([(s * xa, ta + 0.25, za + 0.03), (s * xa, ta - 0.75, za + 0.03), (s * xb, tb - 0.60, zb + 0.03),
                     (s * xb, tb + 0.25, zb + 0.03), (s * xa, ta + 0.25, za + 0.10), (s * xa, ta - 0.75, za + 0.09),
                     (s * xb, tb - 0.60, zb + 0.09), (s * xb, tb + 0.25, zb + 0.10)], (0.82, 0.83, 0.86, 1))
    _add(ext, mp, 'pannelli')
    le_t, te_t, zb_t, tc_t = wing_at(txw)
    _lights(ext, ((-txw - lean - 0.1, le_t - swp - 0.2, zb_t + H * 0.9, LIGHT_RED),
                  (txw + lean + 0.1, le_t - swp - 0.2, zb_t + H * 0.9, LIGHT_GREEN),
                  (0, yt - 0.1, rz * 0.5, (1.0, 1.0, 1.0, 1))), 0.14 if not big else 0.22)

    # motori + piloni
    for (ex, ey, ez, er, el) in g['engines']:
        if g.get('rear'):                        # jet privato: motori ai lati della fusoliera di coda
            _engine(ext, ex, ey, ez, er, el, eng, acc, None)
            mst = Mesh()
            x0, x1 = (rx * 0.6, abs(ex)) if ex > 0 else (-abs(ex), -rx * 0.6)
            mst.box(x0, x1, ey - el * 0.3, ey + el * 0.3, ez - er * 0.35, ez + er * 0.2, tuple(body))
            _add(ext, mst, 'attacco')
            continue
        la, ta, zb, tc = wing_at(abs(ex))
        ey = la + 0.12 * el                      # la gondola sta appesa esattamente sotto l'ala
        ez = zb - 1.07 * er - 0.25
        _engine(ext, ex, ey, ez, er, el, eng, acc, None)
        pm = SMesh()
        sec = lambda z, y, c: dict(le=(ex, y, z), c=c, tc=0.10, n=(1, 0, 0), b=(0, -1, 0), cu=BODY)
        foil_mesh(pm, [sec(ez + er * 0.4, ey + el * 0.18, el * 0.80),
                       sec((ez + er * 0.4 + zb) / 2, ey - el * 0.02, el * 0.66),
                       sec(zb + 0.2, ey - el * 0.12, el * 0.55)], 12, cap=False)
        _add(ext, pm, 'pilone')

    # coda
    hx0, hle0, hte0, hx1, hle1, hte1, hz, hdz = g['htail']
    st = SMesh()
    for s in (-1, 1):
        secs = []
        for f in (0.0, 0.35, 0.7, 1.0):
            x = hx0 + (hx1 - hx0) * f
            le = hle0 + (hle1 - hle0) * f
            te = hte0 + (hte1 - hte0) * f
            c = le - te
            tc = 0.095 - 0.03 * f
            secs.append(dict(le=(s * x, le, hz + hdz * f), c=c, tc=tc, n=(0, 0, 1), b=(0, -1, 0), cu=BODY, cd=GRY))
        foil_mesh(st, secs, 14)
    fy0, fy1, fy2, fy3, fz0, fz1, ft = g['fin']
    secs = []
    for f in (0.0, 0.3, 0.65, 1.0):
        z = fz0 - 0.5 + (fz1 - fz0 + 0.5) * f
        le = fy0 + (fy2 - fy0) * f
        te = fy1 + (fy3 - fy1) * f
        c = le - te
        secs.append(dict(le=(0, le, z), c=c, tc=min(0.14, 2 * ft / c * (1.0 - 0.25 * f)), n=(1, 0, 0), b=(0, -1, 0),
                         cu=ACC))
    foil_mesh(st, secs, 14)
    # deriva dorsale
    foil_mesh(st, [dict(le=(0, fy0 + 3.2, fz0 - 0.2), c=3.4, tc=0.05, n=(1, 0, 0), b=(0, -1, 0), cu=ACC),
                   dict(le=(0, fy0 + 0.1, fz0 + 0.9), c=1.6, tc=0.06, n=(1, 0, 0), b=(0, -1, 0), cu=ACC)], 10)
    _add(ext, st, 'coda')
    _lv0 = livery_of(CUSTOM.get(sp.short, {}))
    dtex = tail_decal_texture(_lv0) if _lv0 else None
    if dtex is not None:                                  # emblema aderente alla sagoma della deriva (non puo uscire)
        zc = (fz0 - 0.5 + fz1) / 2.0
        hf = fz1 - (fz0 - 0.5)
        ys = [fy0, fy2, fy1, fy3]
        y0, y1 = min(ys), max(ys)
        sz = max(hf, y1 - y0)
        yc = (y0 + y1) / 2.0
        outline = []
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            outline.append((fy0 + (fy2 - fy0) * f, f))
        for f in (1.0, 0.75, 0.5, 0.25, 0.0):
            outline.append((fy1 + (fy3 - fy1) * f, f))
        for s in (-1, 1):
            pts, uvs = [], []
            for (yy, f) in outline:
                zz = fz0 - 0.5 + hf * f
                xx = s * (ft * (1.0 - 0.25 * f) + 0.03)
                pts.append((xx, yy, zz))
                u_ = (yy - (yc - sz / 2.0)) / sz                  # stessa immagine su entrambi i lati, senza ribaltarla
                v_ = 1.0 - (zz - (zc - sz / 2.0)) / sz
                uvs.append((u_, v_))
            ext_n = uv_poly(pts, uvs, dtex)
            ext_n.reparentTo(ext)

    # superfici mobili (cuneo): equilibratore, timone, alettoni
    _lv = livery_of(CUSTOM.get(sp.short, {}))
    yh = hle0 + (hte0 - hle0) * 0.72
    yh1 = hle1 + (hte1 - hle1) * 0.72
    zh0, zh1 = hz + 0.05, hz + hdz + 0.05
    parts['elev'] = _ctrl(ext, 'elev', (0, yh, hz + hdz * 0.5),
                          [(-hx1 * 0.97, yh1, zh1 - 0.03), (hx1 * 0.97, yh1, zh1 - 0.03), (hx1 * 0.97, hte1, zh1 + 0.0), (-hx1 * 0.97, hte1, zh1 + 0.0)],
                          [(-hx1 * 0.97, yh1, zh1 + 0.10), (hx1 * 0.97, yh1, zh1 + 0.10), (hx1 * 0.97, hte1, zh1 + 0.01), (-hx1 * 0.97, hte1, zh1 + 0.01)],
                          BODY)
    zr0, zr1 = fz0 + 0.5, fz1 - 0.4
    fr0 = fy0 + (fy1 - fy0) * 0.70
    fr1 = fy2 + (fy3 - fy2) * 0.70
    ft_ = ft * 0.9
    parts['rud'] = _ctrl(ext, 'rud', (0, (fr0 + fr1) / 2, (zr0 + zr1) / 2),
                         [(-ft_, fr0, zr0), (ft_, fr0, zr0), (0.02, fy1 - 0.02, zr0), (-0.02, fy1 - 0.02, zr0)],
                         [(-ft_ * 0.7, fr1, zr1), (ft_ * 0.7, fr1, zr1), (0.02, fy3 - 0.02, zr1), (-0.02, fy3 - 0.02, zr1)],
                         (tuple(_lv['tail2']) + (1.0,)) if _lv else ACC)
    for s, key in ((1, 'ail_r'), (-1, 'ail_l')):
        xa0, xa1 = txw * 0.62, txw * 0.92
        la0, ta0, za0, tca0 = wing_at(xa0)
        la1, ta1, za1, tca1 = wing_at(xa1)
        h0, h1 = ta0 + 0.22 * (la0 - ta0), ta1 + 0.22 * (la1 - ta1)
        ztop0, ztop1 = za0 + tca0 * (la0 - ta0) * 0.5, za1 + tca1 * (la1 - ta1) * 0.5
        sx = lambda v: s * v
        parts[key] = _ctrl(ext, key, (0, (h0 + h1) / 2, (ztop0 + ztop1) / 2),
                           [(sx(xa0), h0, ztop0 - 0.03), (sx(xa1), h1, ztop1 - 0.03), (sx(xa1), ta1, ztop1 - 0.01), (sx(xa0), ta0, ztop0 - 0.01)],
                           [(sx(xa0), h0, ztop0 + 0.05), (sx(xa1), h1, ztop1 + 0.04), (sx(xa1), ta1, ztop1 + 0.0), (sx(xa0), ta0, ztop0 + 0.0)],
                           ACC if not big else BODY)

    # ---- carrello (invariato): gamba + ammortizzatore + ruote con cerchioni
    gear = ext.attachNewNode('carrello')
    gz, wr = g['gear_z'], g['wheel_r']
    pz = -rz * 0.5
    leg = pz - (gz + wr)
    nn = gear.attachNewNode('g_nose')
    nn.setPos(0, sp.wheels[0][1].y, pz)
    m = Mesh()
    m.box(-0.10, 0.10, -0.10, 0.10, -leg * 0.55, 0, GREY)
    m.box(-0.065, 0.065, -0.065, 0.065, -leg, -leg * 0.5, CHROME)
    m.box(-0.05, 0.05, 0.10, 0.45, -leg * 0.7, -leg * 0.5, GREY)
    m.box(-0.12, 0.12, 0.12, 0.20, -leg * 0.35, -leg * 0.25, (1, 1, 0.9, 1))
    _add(nn, m, 'gamba')
    for dx in (-0.3, 0.3):
        _wheel(nn, dx, 0, -leg, wr, 0.12)
    parts['g_nose'] = nn
    mxw, myw = abs(sp.wheels[1][1].x), sp.wheels[1][1].y
    n_tand = 3 if big else 2
    mains = [(1, 'g_r', mxw, myw, n_tand), (-1, 'g_l', mxw, myw, n_tand)]
    if big:
        mains += [(1, 'g_r2', mxw + 6.0, myw + 2.5, 2), (-1, 'g_l2', mxw + 6.0, myw + 2.5, 2)]
    for s, key, gx, gy, nt in mains:
        n = gear.attachNewNode(key)
        n.setPos(s * gx, gy, pz)
        m = Mesh()
        m.box(-0.14, 0.14, -0.14, 0.14, -leg * 0.5, 0, GREY)
        m.box(-0.09, 0.09, -0.09, 0.09, -leg, -leg * 0.45, CHROME)
        m.box(-0.10, 0.10, -0.55 * nt, 0.55 * nt, -leg - 0.03, -leg + 0.12, GREY)
        _beam(m, (0, 0, -leg * 0.15), (-s * 0.5, 0.9, -leg * 0.55), 0.04, GREY)
        _add(n, m, 'gamba')
        for i in range(nt):
            dy = (i - (nt - 1) / 2.0) * 1.1
            for dx in (-0.36, 0.36):
                _wheel(n, dx, dy, -leg, wr, 0.14)
        parts[key] = n
    parts['gear'] = gear

    _airliner_interior(root, parts, sp, yoke=('737' in sp.short))
    root.setTwoSided(True)
    return root, parts


# ---------------------------------------------------------------- Concorde
# Ala a delta ogivale (apertura 25,6 m, superficie 358 m2), fusoliera di 61,7 m con muso a ago che si abbassa
# (droop nose), quattro Olympus 593 in due gondole sotto le ali, grande deriva e niente piano di coda orizzontale.
CONC_LE = [(1.2, 9.6), (1.8, 6.2), (2.6, 2.8), (3.6, -0.4), (4.9, -3.9), (6.4, -7.2), (8.0, -10.2), (9.6, -12.9),
           (11.2, -14.9), (12.2, -15.9), (12.8, -16.3)]
CONC_TE = -18.7
CONC_WZ = -0.9


def concorde_texture(sp):
    if 'concorde' in _TEX_CACHE:
        return _TEX_CACHE['concorde']
    g = sp.geo
    yn, yt, rx, rz, nl, tl = g['fus']
    L = yn - yt
    body = _col255(g['body'])
    acc = _col255(g['accent'])
    K = TEX_K[0]
    cu = CUSTOM.get('Concorde', {})
    P = _Painter(L, (rx, rz), 4096 * K, 1024 * K, K, body)
    glass = (18, 26, 38, 255)
    frame = (205, 208, 214, 255)
    # filetto blu sottile sotto i finestrini
    top, bot = [], []
    for i in range(25):
        x = 11.5 + (L - 13.0) * i / 24
        top.append((x, -0.17))
        bot.append((x, -0.23))
    P.poly(top + bot[::-1], acc + (255,))
    for x in np.arange(2.0, L - 1.0, 3.2):
        P.line_x(x, (110, 114, 124, 55))
    for a in (0.85, -0.65):
        P.line_a(a, 2.0, L - 2.0, (110, 114, 124, 50))
    # finestrini piccoli e ovali
    a = math.asin(0.30 / rz)
    x = 12.6
    while x < L - tl - 2.5:
        if abs(x - 17.0) > 0.9 and abs(x - (L - tl - 6.0)) > 0.9:
            P.rect(x, a, 0.13, 0.19, glass, outline=frame, rad=0.06, width=1)
        x += 0.62
    for dd in (17.0, L - tl - 6.0):
        P.rect(dd, math.asin(-0.15 / rz), 0.80, 1.65, (0, 0, 0, 0), outline=(120, 124, 134, 255), rad=0.15, width=2)
        P.rect(dd, math.asin(0.55 / rz), 0.18, 0.26, glass, outline=frame, rad=0.06)
    # cabina di pilotaggio: parabrezza e finestrini laterali sul muso stretto
    for sd in (0, 1):
        P.poly([(6.7, 1.52), (6.7, 1.05), (8.3, 0.98), (8.4, 1.52)], glass, sides=(sd,), outline=frame)
        P.poly([(8.5, 0.98), (9.6, 0.90), (9.7, 1.38), (8.6, 1.50)], glass, sides=(sd,), outline=frame)
    P.text(cu.get('name') or 'CONCORDE', 19.5, math.asin(0.95 / rz), 0.34, acc)
    P.text(cu.get('reg') or 'SST 001', L - tl - 3.0, math.asin(0.65 / rz), 0.22, (60, 62, 70))
    tex = _make_tex(P.finish(), 'concorde_livrea')
    _TEX_CACHE['concorde'] = tex
    return tex


def _gear_set(ext, parts, sp, pz, gz, wr, n_tand, big):
    """Carrello: gamba, ammortizzatore cromato, ruote con cerchioni (stessa struttura dei jet di linea)."""
    gear = ext.attachNewNode('carrello')
    leg = pz - (gz + wr)
    nn = gear.attachNewNode('g_nose')
    nn.setPos(0, sp.wheels[0][1].y, pz)
    m = Mesh()
    m.box(-0.10, 0.10, -0.10, 0.10, -leg * 0.55, 0, GREY)
    m.box(-0.065, 0.065, -0.065, 0.065, -leg, -leg * 0.5, CHROME)
    m.box(-0.05, 0.05, 0.10, 0.45, -leg * 0.7, -leg * 0.5, GREY)
    m.box(-0.12, 0.12, 0.12, 0.20, -leg * 0.35, -leg * 0.25, (1, 1, 0.9, 1))
    _add(nn, m, 'gamba')
    for dx in (-0.3, 0.3):
        _wheel(nn, dx, 0, -leg, wr, 0.12)
    parts['g_nose'] = nn
    mxw, myw = abs(sp.wheels[1][1].x), sp.wheels[1][1].y
    for s, key in ((1, 'g_r'), (-1, 'g_l')):
        n = gear.attachNewNode(key)
        n.setPos(s * mxw, myw, pz)
        m = Mesh()
        m.box(-0.14, 0.14, -0.14, 0.14, -leg * 0.5, 0, GREY)
        m.box(-0.09, 0.09, -0.09, 0.09, -leg, -leg * 0.45, CHROME)
        m.box(-0.10, 0.10, -0.55 * n_tand, 0.55 * n_tand, -leg - 0.03, -leg + 0.12, GREY)
        _beam(m, (0, 0, -leg * 0.15), (-s * 0.5, 0.9, -leg * 0.55), 0.05, GREY)
        _add(n, m, 'gamba')
        for i in range(n_tand):
            dy = (i - (n_tand - 1) / 2.0) * 1.2
            for dx in (-0.36, 0.36):
                _wheel(n, dx, dy, -leg, wr, 0.14)
        parts[key] = n
    parts['gear'] = gear


def build_concorde_model(sp):
    g = sp.geo
    parts = {}
    root = NodePath('aereo')
    ext = root.attachNewNode('ext')
    parts['ext'] = ext
    body, acc = np.asarray(g['body']), np.asarray(g['accent'])
    GRY = np.asarray((0.80, 0.81, 0.85, 1.0))
    yn, yt, rx, rz, nl, tl = g['fus']
    L = yn - yt
    S = 40

    # ---- fusoliera: muso a ago (sotto piatto, sopra che scende), cilindro stretto, coda rialzata
    ysplit = yn - 11.5
    ts = [max(float(t), 0.003) for t in np.linspace(0, 1, 22) ** 1.2]
    ts = sorted(set([round(t, 6) for t in ts] + [round(11.5 / nl, 6)]))
    rows = []
    for t in ts:
        f = t ** 1.45
        bot = -rz + 0.30 * rz * (1 - t) ** 4
        hh = max(rz * f, 0.003)
        rows.append((yn - nl * t, max(rx * f, 0.003), hh, hh, bot + hh))
    rows.append((yn - nl - (L - nl - tl) * 0.5, rx, rz, rz, 0.0))
    rows.append((yt + tl, rx, rz, rz, 0.0))
    for t in np.linspace(0, 1, 12)[1:]:
        f = 0.06 + 0.94 * (1 - t ** 1.5) ** 0.9
        rows.append((yt + tl * (1 - t), rx * f, rz * f, rz * f, rz * 0.40 * t ** 1.8))
    ys_ = np.array([r[0] for r in rows])
    grid = np.stack([super_ring(w, at, ab, zc, y, S, 2.3) for (y, w, at, ab, zc) in rows])
    k = int(np.argmin(np.abs(ys_ - ysplit)))
    uvv = np.linspace(0, 1, S + 1)
    tex = concorde_texture(sp)
    piv = np.array([0.0, ysplit, -0.6])
    nose = ext.attachNewNode('nose')                 # il muso ruota sulla cerniera (droop nose)
    nose.setPos(*piv)
    sm = SMesh()
    sm.grid(grid[:k + 1] - piv, (1, 1, 1, 1), uv=((yn - ys_[:k + 1]) / L, uvv))
    _add(nose, sm, 'muso').setTexture(tex)
    parts['nose'] = nose
    sm = SMesh()
    sm.grid(grid[k:], (1, 1, 1, 1), uv=((yn - ys_[k:]) / L, uvv))
    _add(ext, sm, 'fusoliera').setTexture(tex)

    # ---- ala a delta ogivale
    sw = SMesh()
    for s in (-1, 1):
        secs = []
        for x, le in CONC_LE:
            c = le - CONC_TE
            u = x / 12.8
            secs.append(dict(le=(s * x, le, CONC_WZ), c=c, tc=0.034 + 0.016 * u, n=(0, 0, 1), b=(0, -1, 0),
                             cu=body, cd=GRY))
        foil_mesh(sw, secs, 26)
    _add(ext, sw, 'ala')

    # ---- gondole motore: due Olympus affiancati sotto ciascuna ala, prese d'aria e ugelli
    nm = SMesh()
    dark = np.asarray((0.12, 0.12, 0.14, 1.0))
    metal = np.asarray((0.34, 0.34, 0.38, 1.0))
    zc = -1.95
    for s in (-1, 1):
        cx = s * 5.8
        outer = [(-3.5, 1.50, 0.78), (-3.7, 1.55, 0.82), (-6.0, 1.55, 0.85), (-13.0, 1.55, 0.85), (-18.2, 1.42, 0.80),
                 (-19.0, 1.28, 0.72)]
        rr = np.stack([super_ring(w, h, h, zc, y, 32, 3.4) for y, w, h in outer])
        rr[..., 0] += cx
        nm.grid(rr, body)
        inner = [(-3.52, 1.42, 0.70), (-4.4, 1.36, 0.66), (-6.2, 1.34, 0.64), (-6.25, 0.10, 0.05)]
        ri = np.stack([super_ring(w, h, h, zc, y, 32, 3.4) for y, w, h in inner])
        ri[..., 0] += cx
        nm.grid(ri, dark)
        ang = np.linspace(0, 2 * math.pi, 25)
        for dx in (-0.72, 0.72):                            # ugelli
            nz = [(-19.0, 0.62), (-20.6, 0.58), (-21.8, 0.50), (-21.85, 0.02)]
            rg = np.stack([np.stack([cx + dx + r * np.cos(ang), np.full_like(ang, y), zc + r * np.sin(ang)], 1)
                           for y, r in nz])
            nm.grid(rg, metal)
    _add(ext, nm, 'motori')
    mm = Mesh()
    for s in (-1, 1):
        mm.box(s * 5.8 - 0.03, s * 5.8 + 0.03, -6.2, -3.55, zc - 0.72, zc + 0.72, body)       # divisorio delle prese
    _add(ext, mm, 'divisori')

    # ---- deriva e timone
    fy0, fy1, fy2, fy3, fz0, fz1 = -11.5, -25.0, -21.0, -26.5, 1.0, 7.3
    st = SMesh()
    secs = []
    for f in (0.0, 0.3, 0.65, 1.0):
        z = fz0 - 0.4 + (fz1 - fz0 + 0.4) * f
        le = fy0 + (fy2 - fy0) * f
        te = fy1 + (fy3 - fy1) * f
        c = le - te
        secs.append(dict(le=(0, le, z), c=c, tc=0.050 - 0.012 * f, n=(1, 0, 0), b=(0, -1, 0), cu=acc))
    foil_mesh(st, secs, 18)
    _add(ext, st, 'deriva')
    fr0 = fy0 + (fy1 - fy0) * 0.70
    fr1 = fy2 + (fy3 - fy2) * 0.70
    zr0, zr1 = fz0 + 0.8, fz1 - 0.5
    parts['rud'] = _ctrl(ext, 'rud', (0, (fr0 + fr1) / 2, (zr0 + zr1) / 2),
                         [(-0.30, fr0, zr0), (0.30, fr0, zr0), (0.03, fy1 - 0.03, zr0), (-0.03, fy1 - 0.03, zr0)],
                         [(-0.14, fr1, zr1), (0.14, fr1, zr1), (0.03, fy3 - 0.03, zr1), (-0.03, fy3 - 0.03, zr1)], acc)

    # ---- elevoni sul bordo d'uscita: interni = profondita' (elev), esterni = rollio (ail_r / ail_l)
    yh = CONC_TE + 2.3
    wt = 0.11
    def elevon(x0, x1):
        return ([(x0, yh, CONC_WZ - wt), (x1, yh, CONC_WZ - wt), (x1, CONC_TE, CONC_WZ - 0.01), (x0, CONC_TE, CONC_WZ - 0.01)],
                [(x0, yh, CONC_WZ + wt), (x1, yh, CONC_WZ + wt), (x1, CONC_TE, CONC_WZ + 0.01), (x0, CONC_TE, CONC_WZ + 0.01)])
    en = ext.attachNewNode('elev')
    en.setPos(0, yh, CONC_WZ)
    mm = Mesh()
    for s in (-1, 1):
        b_, t_ = elevon(*sorted((s * 1.8, s * 6.0)))
        rel = lambda p: (p[0], p[1] - yh, p[2] - CONC_WZ)
        mm.hull([rel(p) for p in b_] + [rel(p) for p in t_], body)
    _add(en, mm, 'elev_g')
    parts['elev'] = en
    for s, key in ((1, 'ail_r'), (-1, 'ail_l')):
        b_, t_ = elevon(*sorted((s * 6.2, s * 11.2)))
        parts[key] = _ctrl(ext, key, (0, yh, CONC_WZ), b_, t_, body)

    _lights(ext, ((-12.9, -17.6, CONC_WZ + 0.1, LIGHT_RED), (12.9, -17.6, CONC_WZ + 0.1, LIGHT_GREEN),
                  (0, yt - 0.1, 0.6, (1.0, 1.0, 1.0, 1))), 0.16)

    _gear_set(ext, parts, sp, -rz * 0.5, g['gear_z'], g['wheel_r'], 2, False)
    _airliner_interior(root, parts, sp, yoke=True)
    root.setTwoSided(True)
    return root, parts


# ---------------------------------------------------------------- specifiche aerei
G = 9.81
FLAPS = (0.0, 0.33, 0.66, 1.0)


def make_specs():
    cessna = Obj(
        name='Cessna 172', short='Cessna 172', kind='prop',
        desc='Monomotore leggero a elica.\nDecollo ~100 km/h, crociera ~200 km/h.\nIl piu agile: ideale per imparare.',
        mass=1100.0, S=16.2, I=(1825.0, 1285.0, 2670.0), T0=3000.0, P=80000.0,
        cd0=0.030, cd_gear=0.012, cd_flap=0.045, k_ind=0.065, cl0=0.25, cl_flap=0.35, cla=4.6,
        a_s=0.28, a0=0.06, Ma=8.0, auth=0.2, Rail=3.2, Lb=2.0, Nb=3.5, Nr=1.8, ref=50.0,
        max_bank=math.radians(55), max_gamma=math.radians(14), max_desc=math.radians(26), vr=28.0, vair=48.0, air_thr=0.55,
        vmo=100.0, gear_time=5.0, stat=0.15, crash_g=7.0, mu_brake=0.55, flaps0=0, rev=0.25,
        wheels=(('nose', Vec3(0, 1.9, -1.5)), ('left', Vec3(-1.35, -0.35, -1.5)), ('right', Vec3(1.35, -0.35, -1.5))),
        points=(('il muso', Vec3(0, 2.8, 0)), ('la coda', Vec3(0, -5.8, 0.25)),
                ("l'ala sinistra", Vec3(-5.5, 0.2, 1.1)), ("l'ala destra", Vec3(5.5, 0.2, 1.1)),
                ('la pancia', Vec3(0, 0.3, -0.7)), ('la pancia', Vec3(0, -2.5, -0.4))),
        cam=(22.0, 6.0), eye=Vec3(0, 0.35, 0.64), tower_fov=40, builder=build_cessna_model, geo=None)

    def jet(name, short, desc, mass, S, I, T0, Ma, Rail, vr, vair, air_thr, vmo, stat, bank, gam, geo,
            wheels, points, cam, eye, cd0=0.022):
        return Obj(name=name, short=short, kind='jet', desc=desc, mass=mass, S=S, I=I, T0=T0, P=0.0,
                   cd0=cd0, cd_gear=0.015, cd_flap=0.09, k_ind=0.045, cl0=0.2, cl_flap=0.9, cla=5.0,
                   a_s=0.25, a0=0.04, Ma=Ma, auth=0.15, Rail=Rail, Lb=1.0, Nb=2.5, Nr=1.0, ref=90.0,
                   max_bank=math.radians(bank), max_gamma=math.radians(gam), max_desc=math.radians(gam * 1.6), vr=vr, vair=vair,
                   air_thr=air_thr, vmo=vmo, gear_time=9.0, stat=stat, crash_g=4.5, mu_brake=0.45,
                   flaps0=2, rev=0.45, wheels=wheels, points=points, cam=cam, eye=eye, tower_fov=30,
                   builder=build_airliner_model, geo=geo)

    b737 = jet('Boeing 737-800', 'Boeing 737-800',
               'Bimotore a getto di linea (low cost).\nDecollo ~280 km/h, crociera ~650 km/h.\nPista lunga: usa i flap 2 al decollo.',
               68000.0, 125.0, (4.0e6, 2.5e6, 5.0e6), 210000.0, 2.5, 1.2, 78.0, 110.0, 0.30, 260.0, 0.25, 47, 10,
               dict(body=(0.94, 0.94, 0.96, 1), accent=(0.10, 0.28, 0.62, 1), engine=LGREY,
                    fus=(18.5, -21.0, 1.9, 2.0, 5.0, 9.0), wing=(1.7, 3.0, -3.8, 17.9, -5.2, -6.8, -1.2, 0.35, 0.7),
                    engines=[(-5.6, 3.6, -2.35, 1.0, 4.4), (5.6, 3.6, -2.35, 1.0, 4.4)],
                    htail=(0.8, -17.0, -19.7, 7.2, -20.0, -21.6, 1.2, 0.3),
                    fin=(-13.5, -20.0, -19.0, -21.5, 1.2, 8.0, 0.25), gear_z=-3.9, wheel_r=0.5),
               (('nose', Vec3(0, 13.0, -3.9)), ('left', Vec3(-2.9, -1.2, -3.9)), ('right', Vec3(2.9, -1.2, -3.9))),
               (('il muso', Vec3(0, 18.3, -0.5)), ('la coda', Vec3(0, -20.0, -0.4)),
                ("l'ala sinistra", Vec3(-17.9, -5.2, -0.5)), ("l'ala destra", Vec3(17.9, -5.2, -0.5)),
                ('il motore sinistro', Vec3(-5.6, 3.6, -3.3)), ('il motore destro', Vec3(5.6, 3.6, -3.3)),
                ('la pancia', Vec3(0, 5.0, -2.0)), ('la pancia', Vec3(0, -9.0, -1.6))),
               (48.0, 15.0), Vec3(0, 15.1, 1.25))

    a320 = jet('Airbus A320', 'Airbus A320',
               'Bimotore a getto di linea, il piu diffuso\nnelle compagnie economiche. Decollo ~270 km/h,\ncrociera ~640 km/h. Fly-by-wire morbido.',
               64000.0, 122.6, (3.8e6, 2.4e6, 4.8e6), 205000.0, 2.6, 1.4, 75.0, 105.0, 0.28, 260.0, 0.25, 47, 10,
               dict(body=(0.95, 0.95, 0.97, 1), accent=(0.95, 0.50, 0.10, 1), engine=LGREY,
                    fus=(17.2, -20.4, 1.95, 2.0, 4.8, 9.0), wing=(1.8, 2.5, -4.0, 17.9, -5.5, -7.0, -1.2, 0.35, 0.8),
                    engines=[(-5.7, 3.0, -2.4, 1.05, 4.0), (5.7, 3.0, -2.4, 1.05, 4.0)],
                    htail=(0.8, -16.8, -19.3, 6.2, -19.8, -21.0, 1.1, 0.3),
                    fin=(-13.0, -19.7, -18.5, -20.6, 1.1, 7.6, 0.25), gear_z=-3.9, wheel_r=0.5),
               (('nose', Vec3(0, 12.2, -3.9)), ('left', Vec3(-2.8, -1.0, -3.9)), ('right', Vec3(2.8, -1.0, -3.9))),
               (('il muso', Vec3(0, 16.6, -0.5)), ('la coda', Vec3(0, -19.6, -0.4)),
                ("l'ala sinistra", Vec3(-17.9, -5.5, -0.4)), ("l'ala destra", Vec3(17.9, -5.5, -0.4)),
                ('il motore sinistro', Vec3(-5.7, 3.0, -3.4)), ('il motore destro', Vec3(5.7, 3.0, -3.4)),
                ('la pancia', Vec3(0, 4.0, -2.0)), ('la pancia', Vec3(0, -9.0, -1.6))),
               (46.0, 15.0), Vec3(0, 14.2, 1.25))

    a380 = jet('Airbus A380', 'Airbus A380',
               'Quadrimotore a due ponti, 560 tonnellate.\nDecollo ~300 km/h, pista lunga necessaria.\nPesante e lento a rispondere: anticipa i comandi.',
               500000.0, 845.0, (3.0e8, 1.0e8, 4.0e8), 1250000.0, 1.6, 0.75, 80.0, 100.0, 0.30, 270.0, 0.30, 40, 8,
               dict(body=(0.95, 0.95, 0.97, 1), accent=(0.35, 0.22, 0.55, 1), engine=LGREY,
                    fus=(36.5, -37.5, 3.6, 4.3, 11.0, 16.0),
                    wing=(3.3, 8.0, -8.5, 39.9, -17.0, -21.0, -2.0, 0.9, 1.8),
                    engines=[(-12.0, 6.0, -4.8, 1.7, 7.0), (12.0, 6.0, -4.8, 1.7, 7.0),
                             (-24.5, -2.0, -4.4, 1.7, 7.0), (24.5, -2.0, -4.4, 1.7, 7.0)],
                    htail=(1.5, -29.0, -34.0, 15.0, -37.0, -39.2, 3.2, 0.8),
                    fin=(-22.0, -34.0, -34.0, -38.0, 3.2, 18.0, 0.4), gear_z=-7.4, wheel_r=0.75),
               (('nose', Vec3(0, 27.0, -7.4)), ('left', Vec3(-7.0, -4.0, -7.4)), ('right', Vec3(7.0, -4.0, -7.4))),
               (('il muso', Vec3(0, 36.0, -1.0)), ('la coda', Vec3(0, -36.0, 0.0)),
                ("l'ala sinistra", Vec3(-39.9, -17.0, -0.2)), ("l'ala destra", Vec3(39.9, -17.0, -0.2)),
                ('il motore', Vec3(-24.5, -2.0, -6.1)), ('il motore', Vec3(24.5, -2.0, -6.1)),
                ('la pancia', Vec3(0, 10.0, -4.2)), ('la pancia', Vec3(0, -20.0, -3.4))),
               (100.0, 34.0), Vec3(0, 29.8, 3.3), cd0=0.024)
    conc = jet('Concorde', 'Concorde',
               'Supersonico a delta: 4 motori Olympus, muso a ago\nche si abbassa per vedere la pista. Decollo ~370 km/h\ncon grande angolo, atterraggio veloce e ripido.',
               120000.0, 358.0, (1.4e7, 6.0e6, 1.7e7), 520000.0, 1.3, 1.3, 95.0, 130.0, 0.6, 330.0, 0.30, 45, 10,
               dict(body=(0.96, 0.96, 0.98, 1), accent=(0.08, 0.16, 0.45, 1), engine=LGREY,
                    fus=(33.0, -28.66, 1.42, 1.55, 15.5, 7.0), delta=True, gear_z=-5.0, wheel_r=0.55),
               (('nose', Vec3(0, 23.0, -5.0)), ('left', Vec3(-3.9, -3.5, -5.0)), ('right', Vec3(3.9, -3.5, -5.0))),
               (('il muso', Vec3(0, 32.6, -0.8)), ('la coda', Vec3(0, -28.2, 0.7)),
                ("l'ala sinistra", Vec3(-12.8, -17.5, -0.9)), ("l'ala destra", Vec3(12.8, -17.5, -0.9)),
                ('il motore sinistro', Vec3(-5.8, -12.0, -2.75)), ('il motore destro', Vec3(5.8, -12.0, -2.75)),
                ('la pancia', Vec3(0, 10.0, -1.6)), ('la pancia', Vec3(0, -12.0, -1.4))),
               (95.0, 20.0), Vec3(0, 22.6, 0.55), cd0=0.013)
    conc.builder = build_concorde_model
    conc.cl0, conc.cla, conc.a_s, conc.a0, conc.auth = 0.05, 3.4, 0.45, 0.10, 0.30
    conc.k_ind, conc.cl_flap, conc.cd_flap, conc.crash_g, conc.rev = 0.20, 0.0, 0.01, 4.0, 0.35
    conc.gear_time = 10.0
    pjet = jet('Jet privato', 'Jet privato',
               'Business jet a lungo raggio, motori in coda e\nderiva a T. Rapido e agile, 12 passeggeri.\nSempre personalizzabile nell\'hangar (colori e livree).',
               22000.0, 58.0, (0.9e6, 0.55e6, 1.1e6), 82000.0, 3.0, 1.8, 66.0, 95.0, 0.30, 250.0, 0.22, 55, 14,
               dict(body=(0.95, 0.95, 0.97, 1), accent=(0.10, 0.28, 0.62, 1), engine=LGREY, rear=True,
                    fus=(14.6, -14.6, 1.3, 1.35, 4.8, 7.5), wing=(1.0, 0.6, -3.2, 13.6, -4.4, -5.9, -0.95, 0.3, 0.9),
                    engines=[(-2.35, -8.2, 0.55, 0.62, 3.6), (2.35, -8.2, 0.55, 0.62, 3.6)],
                    htail=(0.3, -12.4, -14.2, 4.2, -13.2, -14.7, 5.3, 0.35),
                    fin=(-7.6, -13.9, -12.9, -14.9, 0.9, 5.5, 0.22), gear_z=-2.55, wheel_r=0.35),
               (('nose', Vec3(0, 9.0, -2.55)), ('left', Vec3(-1.6, -1.2, -2.55)), ('right', Vec3(1.6, -1.2, -2.55))),
               (('il muso', Vec3(0, 14.5, -0.4)), ('la coda', Vec3(0, -14.4, -0.3)),
                ("l'ala sinistra", Vec3(-13.6, -4.4, -0.1)), ("l'ala destra", Vec3(13.6, -4.4, -0.1)),
                ('il motore sinistro', Vec3(-2.35, -8.2, 0.0)), ('il motore destro', Vec3(2.35, -8.2, 0.0)),
                ('la pancia', Vec3(0, 4.0, -1.35)), ('la pancia', Vec3(0, -6.0, -1.2))),
               (34.0, 11.0), Vec3(0, 11.3, 0.62), cd0=0.020)
    def cargo_of(base, name, desc, key, fus_scale=1.0):
        """Cargo: stessa cellula del passeggeri ma senza finestrini, con portellone sul fianco e livrea fissa."""
        d = dict(base.__dict__)
        d['geo'] = dict(base.geo)
        d['name'] = d['short'] = name
        d['desc'] = desc
        d['cargo'] = True
        d['fixed_livery'] = key
        d['mass'] = base.mass * 1.08
        return Obj(**d)
    fedex = cargo_of(a320, 'FedEx Cargo',
                     'Cargo FedEx Express: nessun finestrino, portellone\nmerci sul fianco. Carica pacchi viola e arancio.\nLivrea ufficiale fissa.', 'FedEx')
    dhl = cargo_of(b737, 'DHL Cargo',
                   'Cargo DHL Express: fusoliera gialla e rossa,\nportellone merci sul fianco. Livrea fissa.', 'DHL')
    amzn = cargo_of(b737, 'Amazon Air',
                    'Cargo Amazon Air: bianco con coda Prime\ne scritta amazon. Portellone merci sul fianco.', 'Amazon')
    for _s in (cessna, b737, a320, a380, conc, pjet):
        _s.cargo = False
        _s.fixed_livery = None
    return [cessna, b737, a320, a380, conc, pjet, fedex, dhl, amzn]


SPECS = make_specs()


# ---------------------------------------------------------------- personalizzazione (hangar)
for _sp in SPECS:
    _sp.base = dict(T0=_sp.T0, P=_sp.P, mass=_sp.mass, I=tuple(_sp.I),
                    body=tuple(_sp.geo['body'][:3]) if _sp.geo else (0.93, 0.93, 0.95),
                    accent=tuple(_sp.geo['accent'][:3]) if _sp.geo else (0.74, 0.10, 0.12))
POWER_F = (1.0, 1.25, 1.6)
WEIGHT_F = (1.0, 0.9, 1.15)


def apply_custom(sp):
    """Applica colori, potenza e peso scelti nell'hangar (e invalida la texture della livrea)."""
    b, c = sp.base, CUSTOM.get(sp.short, {})
    lv = livery_of(c) if sp.geo is not None else None
    if sp.geo is not None:
        sp.geo['body'] = tuple((lv['body'] if lv else c.get('body', b['body']))[:3]) + (1,)
        sp.geo['accent'] = tuple((lv['tail'] if lv else c.get('accent', b['accent']))[:3]) + (1,)
    pw, wt = POWER_F[c.get('power', 0)], WEIGHT_F[c.get('weight', 0)]
    sp.T0, sp.P = b['T0'] * pw, b['P'] * pw
    sp.mass = b['mass'] * wt
    sp.I = tuple(i * wt for i in b['I'])
    for k in (sp.short, 'cessna', 'concorde'):
        _TEX_CACHE.pop(k, None)


def default_text(sp):
    if sp.short in AIRLINES:
        return AIRLINES[sp.short][0], 'N%s%d' % (AIRLINES[sp.short][1], 100 + len(sp.short))
    if sp.short == 'Concorde':
        return 'CONCORDE', 'SST 001'
    return '', 'N172SK'


def load_custom():
    try:
        with open(HANGAR_FILE, encoding='utf-8') as f:
            data = json.load(f)
        for k, v in data.items():
            CUSTOM[k] = {kk: (tuple(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}
    except Exception:
        pass
    for sp in SPECS:
        if getattr(sp, 'fixed_livery', None):          # i cargo hanno sempre la livrea vera del corriere
            CUSTOM.setdefault(sp.short, {})['livery'] = LIV_IDX[sp.fixed_livery]
        apply_custom(sp)


def save_custom():
    try:
        with open(HANGAR_FILE, 'w', encoding='utf-8') as f:
            json.dump(CUSTOM, f, indent=1)
    except Exception:
        pass


load_custom()


# ---------------------------------------------------------------- citta, grattacieli e Burj Khalifa
def window_texture(kind):
    """Facciata a vetro: una campata e un piano per ripetizione (kind 1 = vetrata continua del Burj)."""
    key = 'finestre%d' % kind
    if key in _TEX_CACHE:
        return _TEX_CACHE[key]
    S = 128
    arr = np.zeros((S, S, 3), float)
    if kind == 0:
        arr[:] = (200, 203, 208)
        y0, y1, x0, x1 = 26, 108, 14, 114
        top, bot = np.array((158, 184, 214.)), np.array((36, 60, 94.))
    else:
        arr[:] = (206, 214, 228)
        y0, y1, x0, x1 = 6, 118, 4, 124
        top, bot = np.array((176, 204, 232.)), np.array((74, 104, 146.))
    g = np.linspace(0, 1, y1 - y0)[:, None, None]
    arr[y0:y1, x0:x1] = top * (1 - g) + bot * g
    if kind == 0:
        arr[y0:y0 + 3, x0:x1] = (236, 241, 246)
    else:
        arr[:, S // 2 - 1:S // 2 + 1] = (226, 233, 243)
        arr[118:, :] = (186, 196, 212)
    rng = np.random.default_rng(5)
    arr += rng.normal(0, 3.0, (S, S, 1))
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGB').convert('RGBA')
    t = _make_tex(im, key)
    t.setWrapU(Texture.WM_repeat)
    t.setWrapV(Texture.WM_repeat)
    _TEX_CACHE[key] = t
    return t


class CityBuilder:
    BAY = (4.0, 2.2)
    FLOOR = (3.9, 4.2)
    TINTS = ((0.74, 0.84, 0.95), (0.64, 0.76, 0.82), (0.86, 0.86, 0.90), (0.72, 0.74, 0.82),
             (0.93, 0.87, 0.76), (0.60, 0.70, 0.78))

    def __init__(self, terrain):
        self.ter = terrain
        self.sm = [SMesh(), SMesh()]
        self.used = [False, False]
        self.pm = Mesh()            # tetti, antenne
        self.lights = Mesh()        # luci rosse di segnalazione
        self.layers = [Mesh() for _ in range(5)]   # suolo, strade, piazze, acqua, verde (ognuno con il suo depth offset)
        self.win, self.beac, self.street = [], [], []      # luci notturne: finestre, luci rosse, lampioni
        self.nrng = np.random.default_rng(21)

    def ground(self, x, y, r=0.0):
        return min(self.ter.sample(x + dx, y + dy)[0] for dx, dy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)))

    @staticmethod
    def rect(x, y, w, d, ang):
        c, s = math.cos(ang), math.sin(ang)
        return [(x + c * dx - s * dy, y + s * dx + c * dy)
                for dx, dy in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2))]

    def wall(self, kind, a, b, z0, z1, col, out):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 0.5 or z1 - z0 < 0.5:
            return
        nb = max(1, round(L / self.BAY[kind]))
        nf = max(1, round((z1 - z0) / self.FLOOR[kind]))
        P = np.array([[[a[0], a[1], z0], [b[0], b[1], z0]], [[a[0], a[1], z1], [b[0], b[1], z1]]], float)
        UV = np.array([[[0, 0], [nb, 0]], [[0, nf], [nb, nf]]], float)
        self.sm[kind].grid(P, col, uv=UV, closed=False, face=out)
        self.used[kind] = True
        dx, dy = b[0] - a[0], b[1] - a[1]
        nx, ny = -dy / L, dx / L
        if nx * out[0] + ny * out[1] < 0:
            nx, ny = -nx, -ny
        self.windows(a[0] + nx * 0.5, a[1] + ny * 0.5, dx, dy, z0, z1, nb, nf, kind)

    def windows(self, x0, y0, dx, dy, z0, z1, nb, nf, kind):
        """Finestre illuminate di notte (punti luminosi davanti alle campate della facciata)."""
        p = 0.30 if kind == 0 else 0.10
        m = self.nrng.random((nf, nb)) < p
        jj, ii = np.nonzero(m)
        if not len(ii):
            return
        t = (ii + 0.5) / nb
        z = z0 + (jj + 0.42) * (z1 - z0) / nf
        pick = self.nrng.integers(0, 4, len(ii))
        tones = np.array([(1.0, 0.86, 0.55), (1.0, 0.92, 0.75), (0.80, 0.88, 1.0), (1.0, 0.78, 0.45)])
        col = tones[pick]
        arr = np.column_stack((x0 + dx * t, y0 + dy * t, z, col, np.full(len(ii), 0.9)))
        self.win.append(arr)

    def prism(self, kind, corners, z0, z1, col, roof=(0.38, 0.39, 0.42, 1)):
        n = len(corners)
        cx = sum(c[0] for c in corners) / n
        cy = sum(c[1] for c in corners) / n
        for i in range(n):
            a, b = corners[i], corners[(i + 1) % n]
            self.wall(kind, a, b, z0, z1, col, ((a[0] + b[0]) / 2 - cx, (a[1] + b[1]) / 2 - cy, 0.0))
        if roof:
            for i in range(1, n - 1):
                self.pm.tri_flat((corners[0][0], corners[0][1], z1), (corners[i][0], corners[i][1], z1),
                                 (corners[i + 1][0], corners[i + 1][1], z1), roof, (cx, cy, z0))

    def cylinder(self, kind, x, y, r0, r1, z0, z1, col, seg=20, roof=(0.38, 0.39, 0.42, 1)):
        ang = np.linspace(0, 2 * math.pi, seg + 1)
        rows = [[[x + r * math.cos(a), y + r * math.sin(a), z] for a in ang] for z, r in ((z0, r0), (z1, r1))]
        nb = max(seg // 2, round(2 * math.pi * max(r0, r1) / self.BAY[kind]))
        nf = max(1, round((z1 - z0) / self.FLOOR[kind]))
        UV = np.array([[[nb * i / seg, v] for i in range(seg + 1)] for v in (0, nf)], float)
        self.sm[kind].grid(np.array(rows, float), col, uv=UV, closed=True)
        self.used[kind] = True
        nbc, nfc = max(seg // 2, nb), nf
        p = 0.30 if kind == 0 else 0.10
        m = self.nrng.random((nfc, nbc)) < p
        jj, ii = np.nonzero(m)
        if len(ii):
            fz = (jj + 0.42) / nfc
            rr = r0 + (r1 - r0) * fz + 0.5
            th = 2 * math.pi * (ii + 0.5) / nbc
            tones = np.array([(1.0, 0.86, 0.55), (1.0, 0.92, 0.75), (0.80, 0.88, 1.0), (1.0, 0.78, 0.45)])
            col = tones[self.nrng.integers(0, 4, len(ii))]
            self.win.append(np.column_stack((x + rr * np.cos(th), y + rr * np.sin(th), z0 + fz * (z1 - z0), col,
                                             np.full(len(ii), 0.9))))
        if roof:
            for i in range(seg):
                self.pm.tri_flat((x, y, z1), (x + r1 * math.cos(ang[i]), y + r1 * math.sin(ang[i]), z1),
                                 (x + r1 * math.cos(ang[i + 1]), y + r1 * math.sin(ang[i + 1]), z1), roof, (x, y, z0))

    def antenna(self, x, y, z, h, w=0.5):
        self.pm.box(x - w, x + w, y - w, y + w, z, z + h, (0.55, 0.56, 0.6, 1))
        self.lights.box(x - 0.9, x + 0.9, y - 0.9, y + 0.9, z + h, z + h + 1.8, (1.0, 0.08, 0.05, 1))
        self.beac.append((x, y, z + h + 2.0, 1.0, 0.08, 0.05, 1.0))

    def tower(self, x, y, w, d, h, ang, tint, shape):
        hg = self.ground(x, y, max(w, d) * 0.5) - 1.5
        col = tuple(tint) + (1,)
        if shape == 'step' and h > 60:
            h1 = h * 0.6
            self.prism(0, self.rect(x, y, w, d, ang), hg, hg + h1, col)
            self.prism(0, self.rect(x, y, w * 0.66, d * 0.66, ang), hg + h1, hg + h, col)
            add_obstacle(x, y, 0.5 * math.hypot(w, d), hg + h1)
            add_obstacle(x, y, 0.33 * math.hypot(w, d), hg + h)
        elif shape == 'cyl':
            r = min(w, d) * 0.5
            self.cylinder(0, x, y, r, r * 0.82, hg, hg + h, col, 22)
            add_obstacle(x, y, r, hg + h)
        else:
            self.prism(0, self.rect(x, y, w, d, ang), hg, hg + h, col)
            self.pm.box(x - w * 0.18, x + w * 0.18, y - d * 0.18, y + d * 0.18, hg + h, hg + h + 3.5, (0.5, 0.5, 0.54, 1))
            add_obstacle(x, y, 0.5 * math.hypot(w, d), hg + h)
        if h > 70:
            self.antenna(x, y, hg + h + (3.5 if shape == 'box' else 0.0), min(40.0, h * 0.12))

    def disc(self, layer, x, y, r, z, col, seg=48):
        m = self.layers[layer]
        for i in range(seg):
            a0, a1 = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
            m.tri_flat((x, y, z), (x + r * math.cos(a0), y + r * math.sin(a0), z),
                       (x + r * math.cos(a1), y + r * math.sin(a1), z), col, (x, y, z - 1))

    def road(self, layer, x0, y0, x1, y1, width, z, col):
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L * width / 2, dx / L * width / 2
        if layer == 1:
            for k in range(int(L // 40) + 1):
                t = k * 40.0 / L
                for sgn in (-1, 1):
                    self.street.append((x0 + dx * t + nx * sgn * 1.05, y0 + dy * t + ny * sgn * 1.05, z + 7.0,
                                        1.0, 0.72, 0.38, 0.9))
        self.layers[layer].quad([(x0 + nx, y0 + ny, z), (x1 + nx, y1 + ny, z), (x1 - nx, y1 - ny, z),
                                 (x0 - nx, y0 - ny, z)], col, (x0, y0, z - 1))

    def downtown(self, cx, cy, rng, n=9, radius=120.0):
        """Il centro di una cittadina: qualche grattacielo, piu alti verso il mezzo."""
        placed = []
        for _ in range(n * 6):
            if len(placed) >= n:
                break
            a, r = rng.uniform(0, 2 * math.pi), radius * math.sqrt(rng.uniform(0, 1))
            x, y = cx + r * math.cos(a), cy + r * math.sin(a)
            w, d = rng.uniform(22, 40), rng.uniform(22, 40)
            if any(math.hypot(x - px, y - py) < pr + 0.5 * max(w, d) + 8 for px, py, pr in placed):
                continue
            f = 1.0 - r / radius
            h = rng.uniform(45, 85) + 110 * f * rng.uniform(0.3, 1.0)
            self.tower(x, y, w, d, h, rng.choice((0.0, 0.0, 0.3, 0.8)), rng.choice(self.TINTS),
                       rng.choice(('box', 'box', 'step', 'cyl')))
            placed.append((x, y, 0.5 * max(w, d)))
        gh = self.ground(cx, cy, radius)
        self.disc(0, cx, cy, radius + 40, gh + 0.2, (0.52, 0.52, 0.55, 1), 28)
        return radius + 40

    def burj(self, bx, by):
        hg = self.ground(bx, by, 60.0) - 2.0
        silver = (0.88, 0.91, 0.96, 1)
        self.cylinder(1, bx, by, 44, 40, hg, hg + 26, (0.80, 0.83, 0.88, 1), 6)
        add_obstacle(bx, by, 46, hg + 26)
        core = ((26, 190, 24, 21), (190, 330, 21, 18), (330, 460, 18, 15), (460, 560, 15, 12.5), (560, 640, 12.5, 9.5))
        for z0, z1, r0, r1 in core:
            self.cylinder(1, bx, by, r0, r1, hg + z0, hg + z1, silver, 6, roof=None)
        base_h = (130, 195, 260, 325, 390, 455, 515, 570, 615)
        for w in range(3):
            th = math.radians(90 + 120 * w)
            c, s = math.cos(th), math.sin(th)
            prev = 26.0
            for k, hk in enumerate(base_h):
                top = hk + w * 8.0
                L = 44.0 - 3.5 * k
                W = 22.0 - 1.3 * k
                pts = [(bx + c * px - s * py, by + s * px + c * py)
                       for px, py in ((0.0, -W / 2), (L, -W / 2), (L, W / 2), (0.0, W / 2))]
                self.prism(1, pts, hg + prev, hg + top, silver, roof=(0.62, 0.66, 0.72, 1))
                prev = top
        for k, hk in enumerate(base_h):
            add_obstacle(bx, by, 44.0 - 3.5 * k + 3.0, hg + hk + 16.0)
        for z0, z1, r0, r1 in ((640, 700, 9.5, 5.5), (700, 750, 5.5, 3.2), (750, 795, 3.2, 1.5), (795, 828, 1.5, 0.25)):
            self.cylinder(1, bx, by, r0, r1, hg + z0, hg + z1, silver, 12, roof=None)
            add_obstacle(bx, by, max(r0, 2.0), hg + z1)
        self.lights.box(bx - 1.0, bx + 1.0, by - 1.0, by + 1.0, hg + 826, hg + 830, (1.0, 0.08, 0.05, 1))
        self.beac.append((bx, by, hg + 831.0, 1.0, 0.08, 0.05, 1.0))
        return hg

    def metropolis(self, rng):
        R, pitch = 1150.0, 120.0
        z = CITY_H
        self.disc(0, CITY_X, CITY_Y, R, z + 0.05, (0.50, 0.50, 0.52, 1), 64)
        asph = (0.17, 0.17, 0.19, 1)
        for i in range(-10, 11):
            off = i * pitch + pitch / 2
            if abs(off) < R:
                half = math.sqrt(R * R - off * off) - 4
                self.road(1, CITY_X - half, CITY_Y + off, CITY_X + half, CITY_Y + off, 16, z + 0.08, asph)
                self.road(1, CITY_X + off, CITY_Y - half, CITY_X + off, CITY_Y + half, 16, z + 0.08, asph)
        # viale d'onore che porta al Burj Khalifa e anello attorno alla piazza
        self.road(1, CITY_X - R + 40, CITY_Y, CITY_X + R - 40, CITY_Y, 30, z + 0.09, (0.20, 0.20, 0.22, 1))
        # piazza, laghetto e verde attorno al Burj
        self.disc(2, CITY_X, CITY_Y, 300, z + 0.12, (0.80, 0.77, 0.70, 1), 56)
        self.disc(4, CITY_X, CITY_Y, 255, z + 0.14, (0.30, 0.50, 0.24, 1), 56)
        self.disc(2, CITY_X, CITY_Y, 120, z + 0.16, (0.84, 0.82, 0.76, 1), 40)
        self.disc(3, CITY_X + 150, CITY_Y + 70, 95, z + 0.18, (0.12, 0.42, 0.62, 1), 36)
        self.disc(3, CITY_X - 130, CITY_Y - 90, 70, z + 0.18, (0.12, 0.42, 0.62, 1), 32)
        bhg = self.burj(CITY_X, CITY_Y)
        placed = 0
        for i in range(-10, 11):
            ld_pump()
            for j in range(-10, 11):
                x = CITY_X + i * pitch + rng.uniform(-5, 5)
                y = CITY_Y + j * pitch + rng.uniform(-5, 5)
                dc = math.hypot(x - CITY_X, y - CITY_Y)
                if dc > R - 80 or dc < 340 or rng.random() < 0.15:
                    continue
                f = 1.0 - dc / R
                h = 55.0 + 330.0 * f ** 1.5 * rng.uniform(0.35, 1.0)
                if dc < 800 and rng.random() < 0.06:
                    h = rng.uniform(300, 450)
                w, d = rng.uniform(34, 70), rng.uniform(34, 70)
                self.tower(x, y, w, d, h, 0.0, rng.choice(self.TINTS), rng.choice(('box', 'box', 'step', 'cyl')))
                placed += 1
        return placed

    def finish(self, parent):
        out = []
        for k in (0, 1):
            if self.used[k]:
                n = parent.attachNewNode(self.sm[k].node('facciate%d' % k))
                n.setTexture(window_texture(k))
                out.append(n)
        if self.pm.T:
            parent.attachNewNode(self.pm.node('tetti'))
        if self.lights.T:
            ln = parent.attachNewNode(self.lights.node('segnalazioni'))
            ln.setLightOff()
            self.beacons = ln
        for i, m in enumerate(self.layers):
            if m.T:
                n = parent.attachNewNode(m.node('suolo%d' % i))
                n.setDepthOffset(i + 1)
                if i == 3:
                    n.setLightOff()
        if self.win:                                    # luci notturne della citta
            w = np.vstack(self.win)
            print('finestre illuminate:', len(w))
            GLOW.append((make_points(w, 2.6, parent, 'finestre_citta'), 'steady'))
        if self.street:
            GLOW.append((make_points(self.street, 3.2, parent, 'lampioni'), 'steady'))
        if self.beac:
            GLOW.append((make_points(self.beac, 10.0, parent, 'luci_rosse'), 'beacon'))
        return out


# ---------------------------------------------------------------- aereo (fisica)
class Controls:
    elev = ail = rud = 0.0
    brake = False
    reverse = False


class Aircraft:
    def __init__(self, terrain, spec):
        self.terrain = terrain
        self.sp = spec
        self.model, self.parts = spec.builder(spec)
        add_aircraft_glow(self.parts.get('ext', self.model) if isinstance(self.parts, dict) else self.model, spec)
        self._gl = None
        self.prop_angle = 0.0
        self.airbrake = False
        self.scrape = 0.0            # >0 mentre la fusoliera striscia sulla pista (scintille)
        self.crash_vel = Vec3(0, 0, 0)
        self.crash_pos = Vec3(0, 0, 0)
        tail = next((p for n, p in spec.points if n == 'la coda'), None)
        mw = spec.wheels[1][1]
        self.max_ground_pitch = (math.atan2(tail.z - mw.z, mw.y - tail.y) * 0.92) if tail else 0.3
        W = spec.mass * G
        nose, main = spec.wheels[0][1], spec.wheels[1][1]
        sn = -main.y / (nose.y - main.y)
        self.kc = {}
        for name, share in (('nose', sn), ('left', (1 - sn) / 2), ('right', (1 - sn) / 2)):
            k = share * W / spec.stat
            self.kc[name] = (k, 2 * 0.8 * math.sqrt(k * share * spec.mass))
        self.ditched = False
        self.ditch_t = 0.0
        self.slides = self.build_slides()
        self.reset(False, APTS[0])

    def belly_z(self):
        zs = [p.z for n, p in self.sp.points if n == 'la pancia']
        return min(zs) if zs else -0.7

    def float_z(self):
        """Quota del baricentro quando galleggia (la fusoliera affonda un po')."""
        b = -self.belly_z()
        return b - clamp(0.5 * b, 0.35, 2.0)

    def build_slides(self):
        """Scivoli di evacuazione gonfiabili (zattera per il Cessna): nascosti finche non si ammara."""
        sp = self.sp
        orange, dark = (1.0, 0.42, 0.04, 1), (0.85, 0.30, 0.02, 1)
        out = []
        zw = -self.float_z() + 0.05            # livello dell'acqua nel riferimento dell'aereo

        def slab(m, x0, z0, x1, z1, w, t, col):
            y = (-w, w)
            m.hull([(x0, y[0], z0 - t), (x0, y[1], z0 - t), (x1, y[1], z1 - t), (x1, y[0], z1 - t),
                    (x0, y[0], z0), (x0, y[1], z0), (x1, y[1], z1), (x1, y[0], z1)], col)

        if sp.geo:
            yn, yt, rx, rz, nl, tl = sp.geo['fus']
            ys = [yn - nl - 1.5, yt + tl + 2.0]
            if sp.mass > 300000.0:
                ys += [(ys[0] + ys[1]) / 2 + 9.0, (ys[0] + ys[1]) / 2 - 9.0]
            z0 = -0.25 * rz
            for y in ys:
                for sgn in (1, -1):
                    m = Mesh()
                    dz = min(zw - z0, -0.8)
                    run = max(3.0, -dz * 1.7)
                    slab(m, 0.0, 0.0, sgn * run, dz, 0.75, 0.45, orange)
                    slab(m, sgn * run, dz, sgn * (run + 2.4), dz - 0.05, 1.1, 0.45, dark)      # zattera in fondo
                    n = self.parts['ext'].attachNewNode(m.node('scivolo'))
                    n.setPos(sgn * (rx - 0.05), y, z0)
                    n.setTwoSided(True)
                    n.setLightOff()
                    n.hide()
                    out.append(n)
        else:
            m = Mesh()
            m.box(-1.0, 1.0, -1.4, 1.4, -0.12, 0.12, orange)           # zattera accanto alla porta
            m.box(-1.0, 1.0, -1.4, -1.2, 0.12, 0.35, dark)
            m.box(-1.0, 1.0, 1.2, 1.4, 0.12, 0.35, dark)
            n = self.parts['ext'].attachNewNode(m.node('zattera'))
            n.setPos(2.7, 0.3, zw + 0.1)
            n.setLightOff()
            n.hide()
            out.append(n)
        return out

    def ditch(self):
        """Ammaraggio: l'aereo non affonda, resta a galla e (aerei di linea) apre gli scivoli."""
        if self.ditched or self.crashed:
            return
        self.ditched = True
        self.ditch_t = 0.0
        self.throttle = 0.0
        self.gear_water = True
        self.w = [0.0, 0.0, 0.0]
        self.vel = Vec3(self.vel.x * 0.8, self.vel.y * 0.8, max(self.vel.z, -6.0))

    def float_step(self, dt):
        self.ditch_t += dt
        t = self.ditch_t
        v = self.vel
        hv = Vec3(v.x, v.y, 0.0)
        hv *= math.exp(-dt * (0.9 if hv.length() > 8.0 else 0.45))
        if hv.length() < 0.25:
            hv = Vec3(0, 0, 0)
        zt = self.float_z() + 0.12 * math.sin(t * 1.1)
        az = 8.0 * (zt - self.pos.z) - 3.0 * v.z
        vz = v.z + az * dt
        self.vel = Vec3(hv.x, hv.y, vz)
        self.pos = self.pos + self.vel * dt
        # assetto: si raddrizza sul pelo dell'acqua, con un leggero dondolio
        fh = Vec3(self.fwd.x, self.fwd.y, 0.0)
        if fh.length() < 0.1:
            fh = Vec3(self.right.cross(Vec3(0, 0, 1)))
        fh.normalize()
        fd = Vec3(fh.x, fh.y, 0.03 + 0.012 * math.sin(t * 0.9))
        ud = Vec3(0.02 * math.sin(t * 0.7), 0.0, 1.0)
        a = 1.0 - math.exp(-dt * 1.6)
        self.fwd = self.fwd + (fd - self.fwd) * a
        self.fwd.normalize()
        self.up = self.up + (ud - self.up) * a
        self.up = self.up - self.fwd * self.up.dot(self.fwd)
        self.up.normalize()
        self.right = self.fwd.cross(self.up)
        self.on_ground = False
        self.w = [0.0, 0.0, 0.0]

    def reset(self, air, apt, gate=False):
        sp = self.sp
        self.crashed = False
        self.crash_reason = ''
        self.ditched = False
        self.ditch_t = 0.0
        self.occ = 0                     # persone / pacchi a bordo (si scaricano al prossimo attracco)
        self.eng = bool(air)             # si parte con i motori spenti (E per accenderli), in volo sono accesi
        self.spool = 1.0 if air else 0.0
        self.apt = apt
        f = apt.fwd()
        self.fwd = Vec3(f)
        self.right = Vec3(f.y, -f.x, 0)
        self.up = Vec3(0, 0, 1)
        self.w = [0.0, 0.0, 0.0]
        self.flaps = sp.flaps0
        self.gear = True
        self.gear_pos = 0.0          # 0 = carrello giu, 1 = completamente rientrato
        self.on_ground = True
        self.alpha = self.beta = 0.0
        self.stalled = False
        self.load = 1.0
        gz = -sp.wheels[0][1].z
        if air:
            x, y = apt.world(0, -apt.L / 2 - 9000.0)
            self.pos = Vec3(x, y, apt.h + 900.0)
            self.vel = self.fwd * sp.vair
            self.throttle = sp.air_thr
            self.flaps = 0
            self.gear = False
            self.gear_pos = 1.0
            self.on_ground = False
        else:
            x, y = apt.world(0, -apt.L / 2 + 150.0)
            self.pos = Vec3(x, y, apt.h + gz - sp.stat * 0.7)
            self.vel = Vec3(0, 0, 0)
            self.throttle = 0.0
            if gate:                    # parcheggiato al gate: muso verso l'aereoporto-sud, porta sinistra davanti alla passerella
                cg = getattr(sp, 'cargo', False)
                gv = CARGO_V[0] if cg else GATE_V[3]
                rxf = sp.geo['fus'][2] if sp.geo else 1.3
                ys0 = (sp.geo['fus'][0] - sp.geo['fus'][4] - (9.0 if cg else 1.5)) if sp.geo else 0.0
                x, y = apt.world(GATE_U - 17.0 - rxf, gv + ys0)
                f = -apt.fwd()
                self.fwd = Vec3(f)
                self.right = Vec3(f.y, -f.x, 0)
                self.pos = Vec3(x, y, apt.h + gz - sp.stat * 0.7)

    def toggle_engines(self):
        self.eng = not self.eng
        self.throttle = 0.0              # si accende e si spegne al minimo

    def eff_throttle(self):
        """Gas realmente applicato: zero con i motori spenti o ancora in avviamento."""
        return self.throttle * self.spool

    def crash(self, reason):
        if not self.crashed:
            self.crashed = True
            self.crash_reason = reason
            self.crash_vel = Vec3(self.vel)
            self.crash_pos = Vec3(self.pos)
            self.vel = Vec3(0, 0, 0)
            self.w = [0.0, 0.0, 0.0]

    def climb_limit(self):
        """Angolo di salita massimo sostenibile a gas pieno (spinta meno resistenza), in radianti."""
        sp = self.sp
        V = max(self.vel.length(), 20.0)
        rho = 1.225 * math.exp(-self.pos.z / 8500.0)
        dens = rho / 1.225
        if sp.kind == 'prop':
            T = dens ** 0.85 * min(sp.T0, sp.P / max(V, 10.0))
        else:
            T = sp.T0 * dens ** 0.75 * max(0.25, 1 - V / 600.0)
        q = 0.5 * rho * V * V
        W = sp.mass * G
        cl = W / (q * sp.S)
        fl = FLAPS[self.flaps]
        cd = sp.cd0 + sp.cd_gear * (1.0 - self.gear_pos) + sp.cd_flap * fl * fl + sp.k_ind * cl * cl
        return clamp(0.9 * (T - q * sp.S * cd) / W, 0.0, sp.max_gamma)

    def stall_speed(self):
        sp = self.sp
        rho = 1.225 * math.exp(-self.pos.z / 8500.0)
        fl = FLAPS[self.flaps]
        clmax = sp.cl0 + sp.cl_flap * fl + sp.cla * (sp.a_s - 0.03 * fl)
        return math.sqrt(2 * sp.mass * G / (rho * sp.S * clmax))

    def trim_elev(self, bank_cos):
        """Elevatore necessario per il volo livellato alla velocita attuale."""
        sp = self.sp
        V = max(self.vel.length(), 15.0)
        rho = 1.225 * math.exp(-self.pos.z / 8500.0)
        q = 0.5 * rho * V * V
        fl = FLAPS[self.flaps]
        cl_req = sp.mass * G / (q * sp.S * max(bank_cos, 0.5))
        al = (cl_req - sp.cl0 - sp.cl_flap * fl) / sp.cla
        a0 = sp.a0 - 0.03 * fl
        return clamp((al - a0) / sp.auth, -1.0, 1.0)

    def step(self, dt, ctl):
        if self.crashed:
            return
        if self.ditched:
            self.float_step(dt)
            return
        sp = self.sp
        M, S = sp.mass, sp.S
        terrain = self.terrain
        rate = dt / sp.gear_time     # il carrello si muove lentamente
        self.gear_pos += clamp((0.0 if self.gear else 1.0) - self.gear_pos, -rate, rate)
        wx, wy, wz = self.w
        v = self.vel
        rho = 1.225 * math.exp(-self.pos.z / 8500.0)
        V = v.length()
        F = Vec3(0, 0, -G * M)
        acc_p = acc_r = acc_y = 0.0
        fl = FLAPS[self.flaps]
        vb_x, vb_y, vb_z = v.dot(self.right), v.dot(self.fwd), v.dot(self.up)

        alpha = beta = 0.0
        stall_w = 0.0
        a_s = sp.a_s - 0.03 * fl
        if V > 1.0:
            alpha = math.atan2(-vb_z, vb_y)
            beta = math.atan2(vb_x, math.hypot(vb_y, vb_z))
            q = 0.5 * rho * V * V
            al = clamp(alpha, -1.2, 1.2)
            cl_lin = sp.cl0 + sp.cl_flap * fl + sp.cla * clamp(al, -a_s, a_s)
            stall_w = smoothstep(a_s, a_s + 0.15, abs(al))
            cl = (1 - stall_w) * cl_lin + stall_w * 1.1 * math.sin(2 * al)
            cd = (sp.cd0 + sp.cd_gear * (1.0 - self.gear_pos) + sp.cd_flap * fl * fl
                  + (0.04 if ctl.brake and not self.on_ground else 0.0) + (0.05 if self.airbrake else 0.0)
                  + sp.k_ind * cl_lin * cl_lin + stall_w * 1.9 * math.sin(al) ** 2)
            vh = v / V
            ldir = self.up - vh * self.up.dot(vh)
            if ldir.length() > 1e-3:
                ldir.normalize()
                F += ldir * (q * S * cl)
            F += vh * (-q * S * cd)
            F += self.right * (-q * S * 0.6 * beta)
            self.load = (q * S * cl) / (M * G)

            qr = min(2.5, (V / sp.ref) ** 2 * (rho / 1.225))
            qd = clamp(qr, 0.3, 2.5)
            a0 = sp.a0 - 0.03 * fl
            bc = clamp(beta, -0.5, 0.5)
            mq = 1.05 * math.sqrt(sp.Ma)
            acc_p = qr * (-sp.Ma * (al - a0) + sp.Ma * sp.auth * ctl.elev) - mq * qd * wx
            acc_r = qr * (sp.Rail * ctl.ail - sp.Lb * bc) - 3.0 * qd * wy
            acc_y = qr * (-sp.Nb * bc - sp.Nr * ctl.rud) - 2.5 * qd * wz
            if stall_w > 0.05:
                acc_r += stall_w * 2.2 * math.sin(self.prop_angle * 0.31)
        self.alpha, self.beta = alpha, beta
        self.stalled = stall_w > 0.3

        dens = rho / 1.225
        self.spool += clamp((1.0 if self.eng else 0.0) - self.spool, -dt / 9.0, dt / 6.0)   # avvio 6 s, spegnimento 9 s
        thr = self.throttle * self.spool
        if ctl.reverse and self.on_ground:
            T = -sp.rev * sp.T0 * (0.3 + 0.7 * thr) * self.spool
        elif sp.kind == 'prop':
            T = thr * dens ** 0.85 * min(sp.T0, sp.P / max(vb_y, 10.0))
        else:
            T = thr * sp.T0 * dens ** 0.75 * max(0.25, 1 - max(vb_y, 0.0) / 600.0)
        F += self.fwd * T
        if sp.kind == 'prop':
            acc_y += 0.25 * thr / (1 + V / 25.0)

        wvec = self.right * wx + self.fwd * wy + self.up * wz
        tau = Vec3(0, 0, 0)
        total_n = 0.0
        self.on_ground = False
        if self.gear_pos < 0.5:      # ruote considerate in appoggio finche' il carrello e' piu giu che su
            for name, pb in sp.wheels:
                r = self.right * pb.x + self.fwd * pb.y + self.up * pb.z
                wp = self.pos + r
                gh, raw = terrain.sample(wp.x, wp.y)
                pen = gh - wp.z
                if pen <= 0:
                    continue
                if raw < 0.3:
                    self.ditch()
                    return
                self.on_ground = True
                vp = v + wvec.cross(r)
                k, c = self.kc[name]
                n = k * pen - c * vp.z
                if pen > 3 * sp.stat:
                    n += 10.0 * k * (pen - 3 * sp.stat)
                n = max(n, 0.0)
                total_n += n
                d = Vec3(self.fwd.x, self.fwd.y, 0)
                if name == 'nose':
                    d = rot(d, Vec3(0, 0, 1), -ctl.rud * 0.5 / (1 + V / 12.0))
                ln = d.length()
                d = d / ln if ln > 1e-3 else Vec3(0, 1, 0)
                rd = Vec3(d.y, -d.x, 0)
                paved = on_pavement(wp.x, wp.y)
                mu_b = sp.mu_brake if (ctl.brake and name != 'nose') else 0.0
                ff = -n * ((0.025 if paved else 0.08) + mu_b) * math.tanh(vp.dot(d) / 0.6)
                fl_ = -n * (1.0 if paved else 0.75) * math.tanh(vp.dot(rd) / 0.4)
                fv = d * ff + rd * fl_ + Vec3(0, 0, n)
                F += fv
                tau += r.cross(fv)
            acc_p += tau.dot(self.right) / sp.I[0]
            acc_r += tau.dot(self.fwd) / sp.I[1]
            acc_y += tau.dot(self.up) / sp.I[2]
            self.hard_g = total_n / (M * G)      # solo informativo: un atterraggio duro non distrugge piu l'aereo
        self.scrape = max(0.0, self.scrape - dt)
        for name, pb in sp.points:
            r = self.right * pb.x + self.fwd * pb.y + self.up * pb.z
            wp = self.pos + r
            gh, raw = terrain.sample(wp.x, wp.y)
            pen = gh + 0.02 - wp.z
            if pen > 0:
                vp = v + wvec.cross(r)
                bank_ok = self.up.z > 0.3        # ala/motore a terra in atterraggio: solo scintille, mai uno schianto
                if raw >= 0.3 and on_airport(wp.x, wp.y) and -vp.z < 9.0 and bank_ok:
                    # strisciata sulla pista (coda, pancia, motore): scintille e attrito, ma non ci si schianta
                    self.scrape = 0.25
                    n = clamp(pen * M * G / 0.05 - 0.5 * M * vp.z * 40.0, 0.0, 5.0 * M * G)
                    vh = Vec3(vp.x, vp.y, 0)
                    fric = Vec3(0, 0, 0)
                    if vh.length() > 0.5:
                        fric = vh / vh.length() * (-0.45 * n)
                    fv = Vec3(fric.x, fric.y, n)
                    F += fv
                    tq = r.cross(fv)
                    acc_p += tq.dot(self.right) / sp.I[0]
                    acc_r += tq.dot(self.fwd) / sp.I[1]
                    acc_y += tq.dot(self.up) / sp.I[2]
                    continue
                if raw < 0.3:
                    self.ditch()
                else:
                    self.crash('hai toccato il suolo con ' + name)
                return
        for (ox, oy, rad, top) in OBS_GRID.get((int(self.pos.x // OBS_CELL), int(self.pos.y // OBS_CELL)), ()):
            if self.pos.z < top + 3.0 and (self.pos.x - ox) ** 2 + (self.pos.y - oy) ** 2 < (rad + 4.0) ** 2:
                self.crash('hai colpito un edificio')       # grattacieli, torri, hangar
                return

        self.vel = v + F * (dt / M)
        self.pos = self.pos + self.vel * dt
        wx += acc_p * dt
        wy += acc_r * dt
        wz += acc_y * dt
        self.w = [wx, wy, wz]
        om = self.right * wx + self.fwd * wy + self.up * wz
        ang = om.length() * dt
        if ang > 1e-9:
            k = om / om.length()
            self.right = rot(self.right, k, ang)
            self.fwd = rot(self.fwd, k, ang)
            self.up = rot(self.up, k, ang)
            self.fwd.normalize()
            self.up = self.up - self.fwd * self.up.dot(self.fwd)
            self.up.normalize()
            self.right = self.fwd.cross(self.up)

    def set_cockpit(self, on):
        """In vista CABINA si nasconde l'esterno e si mostra l'interno (l'elica resta visibile)."""
        ext, it = self.parts['ext'], self.parts['int']
        if 'prop' in self.parts:
            (self.parts['prop'].hide if on else self.parts['prop'].show)()
        if on:
            ext.hide()
            it.show()
        else:
            ext.show()
            it.hide()

    def quat(self):
        r, f, u = self.right, self.fwd, self.up
        q = Quat()
        q.setFromMatrix(Mat3(r.x, r.y, r.z, f.x, f.y, f.z, u.x, u.y, u.z))
        return q

    def agl(self):
        gh = self.terrain.sample(self.pos.x, self.pos.y)[0]
        return max(0.0, self.pos.z + self.sp.wheels[0][1].z - gh)

    def update_visual(self, dt, ctl):
        self.model.setPos(self.pos)
        self.model.setQuat(self.quat())
        p = self.parts
        if 'prop' in p:
            self.prop_angle += dt * self.spool * (60.0 + 260.0 * self.throttle)
            p['prop'].setR(self.prop_angle * 57.3 % 360)
        e = smoothstep(0.0, 1.0, self.gear_pos)      # il carrello si piega gradualmente
        if 'nose' in p:                              # Concorde: il muso si abbassa con il carrello fuori
            tgt = -(5.0 if (self.on_ground and self.vel.length() > 25.0) else 12.5) * (1.0 - e)
            self.nose_ang = getattr(self, 'nose_ang', 0.0)
            self.nose_ang += (tgt - self.nose_ang) * min(1.0, dt * 1.2)
            p['nose'].setP(self.nose_ang)
        p['g_nose'].setP(90 * e)
        for key, sgn in (('g_r', 1), ('g_l', -1), ('g_r2', 1), ('g_l2', -1)):
            if key in p:
                p[key].setR(90 * sgn * e)
        for node, kind in p.get('ctl', ()):          # volante / sidestick si muovono coi comandi
            if kind == 'yoke':
                node.setR(ctl.ail * 70)
            else:
                node.setR(ctl.ail * 14)
                node.setP(-ctl.elev * 14)
        if p.get('levers') is not None:
            p['levers'].setY(self.sp.eye.y - 0.35 + self.throttle * 0.15)
        if self.gear_pos >= 0.98:
            p['gear'].hide()
        else:
            p['gear'].show()
        p['elev'].setP(-ctl.elev * 22)
        p['ail_r'].setP(-ctl.ail * 20)
        p['ail_l'].setP(ctl.ail * 20)
        p['rud'].setH(clamp(ctl.rud * 60.0, -30.0, 30.0))
        sl = getattr(self, 'slides', None)
        if sl:
            s = smoothstep(0.0, 1.0, (self.ditch_t - 2.0) / 2.5) if self.ditched else 0.0
            for n in sl:
                if s <= 0.001:
                    n.hide()
                else:
                    n.show()
                    n.setScale(s)


# ---------------------------------------------------------------- applicazione
HORIZON = (0.78, 0.86, 0.94)
ZENITH = (0.16, 0.38, 0.78)
CAM_NAMES = ('ESTERNA', 'CABINA', 'TORRE')

HELP = ('COMANDI\n'
        'Mouse        dove vuoi andare\n'
        '             (sx/dx virata, su/giu salita/discesa)\n'
        'E            accendi / spegni i motori\n'
        'Ctrl+B       schermo intero (anche il pulsante in alto a destra)\n'
        'W            acceleratore\n'
        'S            freni\n'
        'P            retromarcia (a terra)\n'
        'G            carrello (solo in volo)\n'
        'R            quota fissa (si/no), in volo a ogni quota\n'
        'F            flap\n'
        'A / D        timone\n'
        'Ctrl         meno gas\n'
        'C            camera\n'
        'K            avanti di 1 ora (giorno = 12 min, notte = 12 min)\n'
        'Clic destro  trascina: guardati attorno (poi torna)\n'
        'Rotellina    avvicina / allontana la visuale\n'
        'T            aeroporto destinazione\n'
        'N            riparti in volo\n'
        'Backspace    riparti da pista    B  dal gate\n'
        'Tab pausa  H aiuto  Esc menu')


# ---------------------------------------------------------------- effetti: schianto, fuoco, fumo, detriti
class _Particle:
    __slots__ = ('np', 'pos', 'vel', 'age', 'life', 's0', 's1', 'c0', 'c1', 'grav', 'drag', 'spin', 'hpr',
                 'rest', 'solid', 'burn', 'add')


def _rv(a, b):
    return random.uniform(a, b)


def _unit():
    while True:
        v = Vec3(_rv(-1, 1), _rv(-1, 1), _rv(-1, 1))
        n = v.length()
        if 0.05 < n <= 1.0:
            return v / n


WIND = Vec3(5.0, 2.0, 0.0)


class Effects:
    MAXP = 650

    def __init__(self, app):
        self.app = app
        self.root = app.render.attachNewNode('effetti')
        m = Mesh()
        m.sphere(0, 0, 0, 1, 1, 1, (1, 1, 1, 1), 8, 5)
        self.sph = NodePath(m.node('fx_sfera'))
        m = Mesh()
        m.box(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, (1, 1, 1, 1))
        self.cube = NodePath(m.node('fx_cubo'))
        # bolle morbide: carte sempre rivolte alla camera con texture sfumata (fuoco liscio, fumo nuvoloso)
        cmk = CardMaker('bolla')
        cmk.setFrame(-1, 1, -1, 1)
        self.card = NodePath(cmk.generate())
        S = 128
        yy, xx = np.mgrid[-1:1:S * 1j, -1:1:S * 1j]
        r = np.sqrt(xx * xx + yy * yy)
        a = np.clip(1.0 - r, 0.0, 1.0)
        rng = np.random.default_rng(21)
        nz = np.asarray(Image.fromarray((rng.random((9, 9)) * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC)) / 255.0
        nz2 = np.asarray(Image.fromarray((rng.random((20, 20)) * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC)) / 255.0
        for name, al in (('fuoco', a ** 1.3), ('fumo', np.clip(a ** 0.8 * (0.55 + 0.6 * nz * 0.7 + 0.3 * nz2) , 0, 1) * np.clip(1.6 * (1 - r), 0, 1))):
            im = np.zeros((S, S, 4), np.uint8)
            im[..., :3] = 255
            im[..., 3] = (np.clip(al, 0, 1) * 255).astype(np.uint8)
            t = _make_tex(Image.fromarray(im, 'RGBA'), 'fx_' + name)
            t.setWrapU(Texture.WM_clamp)
            t.setWrapV(Texture.WM_clamp)
            setattr(self, 'tex_' + name, t)
        cm = CardMaker('lampo')
        cm.setFrame(-2, 2, -2, 2)
        self.flash = app.render2d.attachNewNode(cm.generate())
        self.flash.setTransparency(TransparencyAttrib.MAlpha)
        self.flash.setDepthTest(False)
        self.flash.setDepthWrite(False)
        self.flash.hide()
        self.P = []
        self.decals = []
        self.crash_on = False
        self.t = 0.0
        self.flash_a = 0.0
        self.hidden = False
        self.k = 1.0
        self.shake_amp = 0.0
        self.water = self.building = False
        self.sev = 0.5

    # ---- particelle
    def spawn(self, pos, vel, s0, s1, life, c0, c1, grav=0.0, drag=0.0, add=False, solid=False, spin=0.0, burn=False):
        if len(self.P) >= self.MAXP:
            return None
        p = _Particle()
        n = (self.cube if solid else self.card).copyTo(self.root)
        if not solid:
            n.setLightOff()
            n.setTexture(self.tex_fuoco if add else self.tex_fumo)
            n.setBillboardPointEye()
            n.setTransparency(TransparencyAttrib.MAlpha)
            n.setDepthWrite(False)
            if add:
                n.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha,
                                                  ColorBlendAttrib.OOne))
                n.setBin('fixed', 20)
        p.np = n
        p.pos, p.vel = Vec3(pos), Vec3(vel)
        p.age, p.life = 0.0, life
        p.s0 = tuple(s0) if isinstance(s0, (tuple, list)) else (s0, s0, s0)
        p.s1 = tuple(s1) if isinstance(s1, (tuple, list)) else (s1, s1, s1)
        p.c0, p.c1 = c0, c1
        p.grav, p.drag, p.add = grav, drag, add
        p.solid, p.burn, p.rest = solid, burn, False
        p.spin = Vec3(_rv(-1, 1), _rv(-1, 1), _rv(-1, 1)) * spin if solid else Vec3(0, 0, 0)
        p.hpr = Vec3(_rv(0, 360), _rv(0, 360), _rv(0, 360))
        n.setPos(p.pos)
        n.setScale(*p.s0)
        n.setColorScale(*c0)
        self.P.append(p)
        return p

    def update_particles(self, dt):
        ter = self.app.terrain
        keep = []
        for p in self.P:
            p.age += dt
            if p.age >= p.life:
                p.np.removeNode()
                continue
            u = p.age / p.life
            if not p.rest:
                p.vel.z -= p.grav * dt
                if p.drag:
                    p.vel *= max(0.0, 1.0 - p.drag * dt)
                p.pos += p.vel * dt
                if p.solid:
                    gh = ter.sample(p.pos.x, p.pos.y)[0]
                    if p.pos.z < gh + 0.15:
                        p.pos.z = gh + 0.15
                        if p.vel.length() > 3.0:
                            p.vel.z = abs(p.vel.z) * 0.3
                            p.vel.x *= 0.55
                            p.vel.y *= 0.55
                            p.spin *= 0.5
                        else:
                            p.rest = True
                            p.burn = False
                    p.hpr += p.spin * dt
                    if p.burn and random.random() < dt * 5.0 and len(self.P) < self.MAXP - 60:
                        self.smoke(p.pos, 0.5, 0.6)
            e = math.sqrt(u) if not p.solid else u
            n = p.np
            n.setPos(p.pos)
            n.setScale(*[a + (b - a) * e for a, b in zip(p.s0, p.s1)])
            c = [a + (b - a) * u for a, b in zip(p.c0, p.c1)]
            if not p.solid and not p.add:
                c[3] *= min(1.0, p.age / 0.35)
            elif p.solid and u > 0.93:
                c[3] = 1.0
            n.setColorScale(*c)
            if p.solid:
                n.setHpr(p.hpr.x, p.hpr.y, p.hpr.z)
            keep.append(p)
        self.P = keep

    # ---- ingredienti
    def smoke(self, pos, size=1.0, life=1.0, dark=0.12):
        k = self.k * size
        d = dark
        self.spawn(pos + _unit() * 1.2 * k, Vec3(WIND.x + _rv(-1.2, 1.2), WIND.y + _rv(-1.2, 1.2), _rv(5.0, 9.0)),
                   1.6 * k, _rv(9.0, 13.0) * k, _rv(9.0, 15.0) * life,
                   (d, d * 0.95, d * 0.9, 0.75), (0.34, 0.34, 0.36, 0.0), grav=-0.4, drag=0.12)

    def flame(self, pos, size=1.0):
        k = self.k * size
        self.spawn(pos, Vec3(_rv(-1.5, 1.5), _rv(-1.5, 1.5), _rv(3.0, 7.0)), 2.4 * k, 0.5 * k, _rv(0.6, 1.3),
                   (1.0, _rv(0.55, 0.85), 0.2, 0.85), (0.9, 0.12, 0.0, 0.0), grav=-4.0, drag=0.4, add=True)

    def spark(self, pos, vel, size=0.35, life=None):
        self.spawn(pos, vel, size, size * 0.4, life or _rv(0.4, 1.1), (1.0, 0.85, 0.45, 1.0), (1.0, 0.25, 0.0, 0.0),
                   grav=9.8, drag=0.2, add=True)

    def blast(self, pos, k, sev):
        """Palla di fuoco, fumo nero e anello di polvere."""
        for _ in range(int(20 + 14 * sev)):
            v = _unit()
            v.z = abs(v.z) * 0.9 + 0.1
            sp = _rv(4.0, 24.0) * math.sqrt(k)
            self.spawn(pos + _unit() * 1.5 * k, v * sp, _rv(1.5, 3.2) * k, _rv(7.0, 13.0) * k * (0.7 + 0.35 * sev),
                       _rv(0.8, 2.1), (1.0, _rv(0.8, 0.97), 0.62, 0.95), (0.85, 0.1, 0.0, 0.0), drag=0.7, add=True)
        for _ in range(12):
            v = _unit()
            v.z = abs(v.z)
            self.spawn(pos + _unit() * 2.0 * k, v * _rv(3.0, 14.0), 2.0 * k, _rv(10.0, 16.0) * k, _rv(6.0, 11.0),
                       (0.05, 0.045, 0.04, 0.85), (0.33, 0.33, 0.34, 0.0), grav=-1.0, drag=0.5)
        if not self.water:
            for i in range(20):
                a = 2 * math.pi * i / 20 + _rv(-0.1, 0.1)
                sp = _rv(14.0, 32.0) * math.sqrt(k)
                self.spawn(Vec3(pos.x, pos.y, pos.z + 0.5), Vec3(math.cos(a) * sp, math.sin(a) * sp, _rv(0.5, 3.0)),
                           2.5 * k, _rv(9.0, 14.0) * k, _rv(2.5, 4.5), (0.58, 0.52, 0.42, 0.6), (0.6, 0.56, 0.5, 0.0),
                           drag=0.9)

    def debris(self, pos, cv, k, sev, pieces=None):
        sp = self.app.ac.sp
        body = tuple(sp.geo['body'][:3]) if sp.geo else tuple(sp.base['body'])
        acc = tuple(sp.geo['accent'][:3]) if sp.geo else tuple(sp.base['accent'])
        pal = [body, body, acc, (0.62, 0.64, 0.68), (0.2, 0.2, 0.22), (0.09, 0.08, 0.08), (0.75, 0.76, 0.8)]
        n = pieces if pieces is not None else int(16 + 22 * sev)
        hv = Vec3(cv.x, cv.y, 0) * 0.35
        for i in range(n):
            big = i < n * 0.35
            sz = (_rv(1.2, 4.2) * k, _rv(0.5, 2.6) * k, _rv(0.08, 0.5) * k) if big else \
                 (_rv(0.2, 1.1) * k, _rv(0.2, 0.9) * k, _rv(0.04, 0.3) * k)
            col = random.choice(pal)
            v = _unit() * _rv(4.0, 22.0 + 12 * sev) + hv
            v.z = abs(v.z) * 0.9 + _rv(2.0, 8.0 + 14 * sev)
            self.spawn(pos + _unit() * k, v, sz, sz, _rv(70.0, 110.0), col + (1.0,), col + (1.0,), grav=9.81,
                       drag=0.02, solid=True, spin=_rv(100.0, 520.0), burn=random.random() < 0.4 and not self.water)

    def splash(self, pos, k, sev):
        for _ in range(int(40 + 40 * sev)):
            v = _unit()
            v.z = abs(v.z)
            sp = _rv(4.0, 14.0)
            self.spawn(pos + _unit() * k, Vec3(v.x * sp, v.y * sp, _rv(10.0, 30.0 + 18 * sev)), 1.2 * k,
                       _rv(3.5, 7.0) * k, _rv(1.6, 3.4), (0.92, 0.96, 1.0, 0.85), (0.7, 0.85, 1.0, 0.0), grav=14.0, drag=0.1)
        for i in range(24):
            a = 2 * math.pi * i / 24
            sp = _rv(8.0, 18.0)
            self.spawn(Vec3(pos.x, pos.y, 0.5), Vec3(math.cos(a) * sp, math.sin(a) * sp, 1.0), 2.0 * k,
                       _rv(8.0, 14.0) * k, _rv(3.0, 5.0), (0.9, 0.95, 1.0, 0.6), (0.8, 0.9, 1.0, 0.0), drag=0.7)

    def scorch(self, pos, rx, ry, hdg, a=0.75):
        if len(self.decals) > 60:
            self.decals.pop(0).removeNode()
        n = self.card.copyTo(self.root)
        n.setLightOff()
        n.setTexture(self.tex_fumo)
        n.setTwoSided(True)
        n.setTransparency(TransparencyAttrib.MAlpha)
        n.setDepthWrite(False)
        n.setDepthOffset(3)
        n.setPos(pos.x, pos.y, pos.z + 0.12)
        n.setHpr(hdg, -90, 0)
        n.setScale(rx * 1.3, 1, ry * 1.3)
        n.setColorScale(0.03, 0.028, 0.026, min(1.0, a * 1.3))
        self.decals.append(n)

    def tire_smoke(self, ac):
        sp = ac.sp
        k = clamp((sp.mass / 60000.0) ** 0.3, 0.5, 1.6)
        for name, pb in sp.wheels[1:]:
            wp = ac.pos + ac.right * pb.x + ac.fwd * pb.y + ac.up * pb.z
            for _ in range(7):
                self.spawn(wp + _unit() * 0.4, Vec3(ac.vel.x * 0.15, ac.vel.y * 0.15, 0) + _unit() * 1.4,
                           0.7 * k, _rv(3.0, 5.5) * k, _rv(2.0, 3.6), (0.92, 0.92, 0.92, 0.7), (0.9, 0.9, 0.9, 0.0),
                           drag=0.5)

    def scrape_sparks(self, ac):
        sp = ac.sp
        low = min(sp.points, key=lambda q: (ac.right * q[1].x + ac.fwd * q[1].y + ac.up * q[1].z).z)[1]
        wp = ac.pos + ac.right * low.x + ac.fwd * low.y + ac.up * low.z
        back = Vec3(-ac.vel.x, -ac.vel.y, 0)
        if back.length() > 0.1:
            back.normalize()
        for _ in range(3):
            self.spark(wp, back * _rv(2.0, 10.0) + _unit() * 4.0 + Vec3(0, 0, _rv(1.0, 5.0)), 0.18, _rv(0.25, 0.7))

    # ---- schianto
    def start_crash(self, ac):
        app = self.app
        sp = ac.sp
        self.clear(None)
        self.crash_on = True
        self.t = 0.0
        cv = Vec3(ac.crash_vel)
        speed = cv.length()
        self.k = k = clamp((sp.mass / 60000.0) ** 0.34, 0.45, 2.2)
        self.sev = sev = clamp(speed / 70.0, 0.12, 1.6)
        self.water = 'acqua' in ac.crash_reason
        self.building = 'edificio' in ac.crash_reason
        pos = Vec3(ac.crash_pos)
        gh = app.terrain.sample(pos.x, pos.y)[0]
        self.clr = clamp(pos.z - gh, 0.4, 3.0 * k)
        self.z0 = pos.z
        self.sink = 0.0
        self.slide_dist = 0.0
        self.last_scorch = 0.0
        self.acc_f = self.acc_s = self.acc_d = 0.0
        self.fire = 0.0 if self.water else clamp((speed - 6.0) / 35.0, 0.3, 1.0)
        self.orbit0 = math.atan2(-ac.fwd.y, -ac.fwd.x)
        self.shake_amp = clamp(2.5 * k * sev + 0.6, 0.6, 7.0)
        self.flash_a = 1.0 if sev > 0.35 else 0.35
        self.flash.setColorScale(1.0, 0.92, 0.75, 1.0)
        self.flash.show()
        self.hidden = False
        self.mode = 'slide'
        # assetto del relitto: muso piegato, ala caduta
        if not self.water:
            for sgn in (1, -1):
                f2 = rot(ac.fwd, ac.right, sgn * 0.2)
                if f2.z < ac.fwd.z:
                    break
            ang = sgn * _rv(0.08, 0.38) * (1.0 if ac.fwd.z > -0.25 else 0.3)
            ac.fwd, ac.up = rot(ac.fwd, ac.right, ang), rot(ac.up, ac.right, ang)
            roll = _rv(-0.6, 0.6)
            ac.right, ac.up = rot(ac.right, ac.fwd, roll), rot(ac.up, ac.fwd, roll)
        hv = Vec3(cv.x, cv.y, 0)
        self.wv = Vec3(-ac.fwd.x * 3.0, -ac.fwd.y * 3.0, 0) if self.building else hv * 0.55
        if pos.z - gh > self.clr + 1.5:
            self.mode = 'fall'
        self.blast_pos = Vec3(pos.x, pos.y, max(pos.z, gh + 1.0))
        self.rest_pos = Vec3(pos.x, pos.y, gh)
        disintegrate = sev > 0.95 and not self.water
        self.offs = [Vec3(_rv(-3, 3) * k, _rv(-5, 5) * k, _rv(0.2, 1.5)) for _ in range(4)]
        if self.water:
            self.splash(Vec3(pos.x, pos.y, 0.0), k, sev)
            ac.model.setColorScale(0.55, 0.6, 0.65, 1)
            self.debris(Vec3(pos.x, pos.y, 1.0), cv, k, sev * 0.6, pieces=int(8 + 10 * sev))
            return
        ac.model.setColorScale(0.26, 0.24, 0.23, 1)
        self.blast(self.blast_pos, k, sev)
        self.debris(self.blast_pos, cv, k, sev, pieces=int((16 + 22 * sev) * (1.8 if disintegrate else 1.0)))
        for _ in range(int(30 + 30 * sev)):
            self.spark(self.blast_pos + _unit(), _unit() * _rv(8.0, 30.0) + Vec3(0, 0, 6.0), _rv(0.15, 0.4))
        if disintegrate:
            ac.model.hide()
            self.hidden = True
            self.mode = 'rest'
            self.offs = [Vec3(_rv(-14, 14) * k, _rv(-18, 18) * k, _rv(0.2, 1.5)) for _ in range(7)]
            hd = math.degrees(math.atan2(-cv.x, cv.y)) if hv.length() > 1 else 0.0
            self.scorch(self.rest_pos, 9 * k, 20 * k + speed * 0.3, hd, 0.8)
        elif self.mode == 'slide':
            hd = math.degrees(math.atan2(-cv.x, cv.y)) if hv.length() > 1 else 0.0
            self.scorch(self.rest_pos, 4 * k, 7 * k, hd, 0.7)

    def update_wreck(self, dt):
        app, ac = self.app, self.app.ac
        if self.hidden:
            return
        ter = app.terrain
        if self.water:
            self.sink = min(self.sink + dt * (1.4 if self.t > 1.0 else 0.2), 26.0)
            ac.pos = Vec3(ac.pos.x, ac.pos.y, self.z0 - self.sink)
            if self.sink >= 26.0:
                ac.model.hide()
                self.hidden = True
            return
        if self.mode == 'rest':
            return
        if self.mode == 'fall':
            self.wv.z -= 9.81 * dt
            ac.pos = ac.pos + self.wv * dt
            gh = ter.sample(ac.pos.x, ac.pos.y)[0]
            if ac.pos.z <= gh + self.clr:
                ac.pos = Vec3(ac.pos.x, ac.pos.y, gh + self.clr)
                impact = -self.wv.z
                self.wv = Vec3(self.wv.x * 0.4, self.wv.y * 0.4, 0)
                self.mode = 'slide'
                if impact > 8.0:
                    self.blast(Vec3(ac.pos.x, ac.pos.y, gh + 1.0), self.k * 0.7, clamp(impact / 60.0, 0.2, 1.0))
                    self.flash_a = max(self.flash_a, 0.25)
                    self.flash.show()
                    for _ in range(25):
                        self.spark(ac.pos, _unit() * _rv(6.0, 22.0) + Vec3(0, 0, 5.0), _rv(0.15, 0.4))
                self.scorch(Vec3(ac.pos.x, ac.pos.y, gh), 5 * self.k, 7 * self.k, 0.0, 0.7)
            return
        hs = math.hypot(self.wv.x, self.wv.y)
        if hs > 0.5:
            f = max(0.0, hs - 9.0 * dt) / hs
            self.wv = Vec3(self.wv.x * f, self.wv.y * f, 0)
            x, y = ac.pos.x + self.wv.x * dt, ac.pos.y + self.wv.y * dt
            ac.pos = Vec3(x, y, ter.sample(x, y)[0] + self.clr)
            self.slide_dist += hs * dt
            self.acc_d += dt * hs
            while self.acc_d >= 1.2:
                self.acc_d -= 1.2
                self.spawn(Vec3(x, y, ac.pos.z - self.clr + 0.4) + _unit(), _unit() * 3.0 + Vec3(0, 0, 2.0), 1.6 * self.k,
                           _rv(5.0, 8.0) * self.k, _rv(2.0, 3.5), (0.58, 0.52, 0.42, 0.55), (0.6, 0.56, 0.5, 0.0), drag=0.6)
                if hs > 6.0:
                    self.spark(ac.pos, Vec3(-self.wv.x, -self.wv.y, 0) * 0.3 + _unit() * 5.0 + Vec3(0, 0, 3.0), 0.3)
            if self.slide_dist - self.last_scorch > 6.0:
                self.last_scorch = self.slide_dist
                self.scorch(Vec3(x, y, ac.pos.z - self.clr), 2.5 * self.k, 5.0 * self.k,
                            math.degrees(math.atan2(-self.wv.x, self.wv.y)), 0.55)
        else:
            self.mode = 'rest'
            self.rest_pos = Vec3(ac.pos.x, ac.pos.y, ac.pos.z - self.clr)
            self.scorch(self.rest_pos, 7 * self.k, 10 * self.k, _rv(0, 360), 0.8)

    def emit(self, dt):
        ac = self.app.ac
        base = Vec3(self.rest_pos) if self.hidden else Vec3(ac.pos.x, ac.pos.y, ac.pos.z - self.clr * 0.5)
        if self.water:
            if self.t < 7.0:
                self.acc_s += dt * 4.0
                while self.acc_s >= 1:
                    self.acc_s -= 1
                    self.spawn(Vec3(ac.pos.x, ac.pos.y, 1.0) + _unit() * 2 * self.k, Vec3(_rv(-1, 1), _rv(-1, 1), _rv(1, 3)),
                               2.0 * self.k, _rv(5, 8) * self.k, _rv(3, 5), (0.9, 0.93, 0.97, 0.5), (0.9, 0.93, 0.97, 0.0), drag=0.4)
            return
        inten = self.fire * clamp(1.0 - self.t / 140.0, 0.0, 1.0) ** 0.7
        if inten > 0.02:
            self.acc_f += dt * 24.0 * inten
            while self.acc_f >= 1:
                self.acc_f -= 1
                self.flame(base + random.choice(self.offs) + _unit() * self.k)
        self.acc_s += dt * (2.0 + 9.0 * inten)
        while self.acc_s >= 1:
            self.acc_s -= 1
            self.smoke(base + random.choice(self.offs), 1.0, 1.0, _rv(0.06, 0.2) if inten > 0.3 else _rv(0.25, 0.4))

    def shake(self):
        if not self.crash_on or self.t > 3.5:
            return Vec3(0, 0, 0)
        a = self.shake_amp * math.exp(-self.t * 1.7)
        return Vec3(_rv(-a, a), _rv(-a, a), _rv(-a, a) * 0.6)

    def update(self, dt):
        if self.crash_on:
            self.t += dt
            self.update_wreck(dt)
            self.emit(dt)
        if self.flash_a > 0:
            self.flash_a -= dt * 1.1
            if self.flash_a <= 0:
                self.flash.hide()
            else:
                self.flash.setColorScale(1.0, 0.92, 0.75, min(1.0, self.flash_a) ** 1.5 * 0.9)
        self.update_particles(dt)

    def clear(self, ac):
        for p in self.P:
            p.np.removeNode()
        self.P = []
        for n in self.decals:
            n.removeNode()
        self.decals = []
        self.flash.hide()
        self.flash_a = 0.0
        self.crash_on = False
        self.hidden = False
        if ac is not None:
            ac.model.show()
            ac.model.setColorScale(1, 1, 1, 1)


# ---------------------------------------------------------------- audio sintetizzato (numpy -> WAV)
def _fnoise(n, lo, hi, slope, sr, rng):
    """Rumore filtrato in banda, periodico per costruzione (si puo ripetere senza scatti)."""
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    shape = np.where((f >= lo) & (f <= hi), np.maximum(f, 1.0) ** slope, 0.0)
    x = np.fft.irfft(spec * shape, n)
    return x / (np.max(np.abs(x)) + 1e-9)


class Audio:
    SR = 22050

    def __init__(self, app):
        self.app = app
        self.ok = False
        self.s = {}
        self.vol = {}
        if ARGS.selftest:
            return
        try:
            sr = self.SR
            rng = np.random.default_rng(11)
            d = os.path.join(tempfile.gettempdir(), 'flightworld_audio')
            os.makedirs(d, exist_ok=True)
            t3 = np.arange(sr * 3) / sr
            t2 = np.arange(sr * 2) / sr
            jet = 0.9 * _fnoise(sr * 3, 40, 700, -0.4, sr, rng) + 0.45 * _fnoise(sr * 3, 1500, 7000, 0.0, sr, rng)
            for fr, am in ((1800, 0.18), (2751, 0.12), (3603, 0.08)):
                jet += am * np.sin(2 * np.pi * fr * t3)
            prop = sum((1.0 / k ** 0.8) * np.sin(2 * np.pi * 45 * k * t2 + k) for k in range(1, 15))
            prop = prop * (1 + 0.25 * np.sin(2 * np.pi * 2 * t2)) / 3.0 + 0.35 * _fnoise(sr * 2, 80, 2500, -0.5, sr, rng)
            wind = _fnoise(sr * 3, 150, 3500, -0.8, sr, rng)
            rumble = _fnoise(sr * 3, 15, 400, -0.3, sr, rng) * (0.7 + 0.3 * np.sin(2 * np.pi * 8 * t3))
            tb = np.arange(int(4.0 * sr)) / sr
            boom = (_fnoise(len(tb), 25, 500, -0.3, sr, rng) * np.exp(-tb / 0.9)
                    + 0.9 * _fnoise(len(tb), 500, 6000, 0.0, sr, rng) * np.exp(-tb / 0.07)
                    + 0.8 * np.sin(2 * np.pi * (70 * tb - 8 * tb ** 2)) * np.exp(-tb / 0.8)
                    + 0.5 * _fnoise(len(tb), 20, 200, -0.2, sr, rng) * np.exp(-np.abs(tb - 0.7) / 0.9))
            tt = np.arange(int(0.45 * sr)) / sr
            touch = (_fnoise(len(tt), 150, 3000, -0.2, sr, rng) * np.exp(-tt / 0.1)
                     + 0.3 * np.sin(2 * np.pi * 1400 * tt) * np.exp(-tt / 0.14))
            paths = {}
            for name, x, g in (('jet', jet, 0.8), ('prop', prop, 0.8), ('wind', wind, 0.8), ('rumble', rumble, 0.9),
                               ('boom', boom, 0.98), ('touch', touch, 0.9)):
                x = x / (np.max(np.abs(x)) + 1e-9) * g
                p = os.path.join(d, name + '.wav')
                with wave.open(p, 'wb') as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sr)
                    w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
                paths[name] = p
            self.s = {k: app.loader.loadSfx(Filename.fromOsSpecific(p)) for k, p in paths.items()}
            self.ok = all(s.length() > 0.05 for s in self.s.values())
            if self.ok:
                for k in ('jet', 'prop', 'wind', 'rumble'):
                    self.s[k].setLoop(True)
                    self.s[k].setVolume(0.0)
                    self.s[k].play()
                    self.vol[k] = 0.0
        except Exception as ex:           # senza scheda audio il gioco funziona lo stesso
            print('audio non disponibile:', ex)
            self.ok = False

    def one(self, name, vol=1.0):
        if self.ok:
            self.s[name].setVolume(vol)
            self.s[name].play()

    def crash(self):
        if self.ok:
            for k in ('jet', 'prop', 'wind', 'rumble'):
                self.vol[k] = 0.0
                self.s[k].setVolume(0.0)
            self.one('boom', 1.0)

    def update(self, dt):
        if not self.ok:
            return
        app = self.app
        ac = app.ac
        run = (not app.in_menu) and (not app.paused) and not ac.crashed
        V = ac.vel.length()
        thr = ac.throttle
        prop = ac.sp.kind == 'prop'
        tgt = {'jet': 0.0, 'prop': 0.0, 'wind': 0.0, 'rumble': 0.0}
        rate = 1.0
        if run:
            sp_ = ac.spool                 # avviamento/spegnimento: il rumore sale e scende con i giri
            if prop:
                tgt['prop'] = sp_ * (0.22 + 0.6 * thr)
                rate = 0.3 + sp_ * (0.4 + 0.8 * thr)
            else:
                tgt['jet'] = sp_ * (0.16 + 0.65 * thr)
                rate = 0.3 + sp_ * (0.3 + 0.75 * thr)
            tgt['wind'] = clamp(V / 170.0, 0.0, 1.0) ** 1.2 * 0.75
            if ac.on_ground:
                tgt['rumble'] = clamp(V / 50.0, 0.0, 1.0) * 0.6
        elif app.in_menu and not app.in_hangar:
            tgt['jet' if not prop else 'prop'] = 0.0
        if app.cam_mode == 1:
            for k in tgt:
                tgt[k] *= 0.7
        for k, v in tgt.items():
            self.vol[k] += (v - self.vol[k]) * min(1.0, dt * 5.0)
            self.s[k].setVolume(self.vol[k])
        self.s['jet'].setPlayRate(rate)
        self.s['prop'].setPlayRate(rate)


# ---------------------------------------------------------------- multiplayer (UDP, solo libreria standard)
# Stesso schema della collaborazione di Palco: chi crea la partita e l'"host", gli altri entrano con un
# codice (indirizzo + token), l'host fa da ripetitore e la sessione vive finche il suo gioco e aperto.
NET_PORT = 47800
NET_HZ = 20.0
NET_MAX = 8
CODE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'


def _clean_state(s):
    """Un pacchetto arriva da Internet: si controlla tutto prima di usarlo."""
    try:
        def vec(v, n=3):
            r = [float(x) for x in v[:n]]
            if len(r) != n or not all(math.isfinite(x) and abs(x) < 1e7 for x in r):
                raise ValueError
            return r
        c = s['c']
        col = lambda v: [min(1.0, max(0.0, float(x))) for x in v[:3]]
        out = dict(id=int(s['id']), n=re.sub(r'[^\w .-]', '', str(s['n']))[:14] or 'PILOTA',
                   s=int(s['s']), c=[col(c[0]), col(c[1]), str(c[2])[:18], str(c[3])[:12],
                                     min(len(LIVERIES) - 1, max(0, int(c[4]))) if len(c) > 4 else 0],
                   p=vec(s['p']), f=vec(s['f']), u=vec(s['u']), v=vec(s['v']),
                   g=min(1.0, max(0.0, float(s['g']))), th=min(1.0, max(0.0, float(s['th']))),
                   fl=int(s['fl']) % 4, cr=bool(s['cr']), og=bool(s['og']), hd=bool(s['hd']),
                   ct=vec(s['ct']))
        if not 0 <= out['s'] < len(SPECS):
            return None
        return out
    except Exception:
        return None


class Net:
    def __init__(self):
        self.sock = None
        self.role = None            # None, 'host' o 'client'
        self.name = 'PILOTA'
        self.token = ''
        self.my_id = 0
        self.joined = False
        self.status = ''
        self.peers = {}             # host: indirizzo -> dict(id, name, last)
        self.states = {}            # id -> (stato, istante di ricezione)
        self.host_addr = None
        self.pub_ip = None
        self.upnp = ''
        self.upnp_ctl = None
        self.port = NET_PORT
        self.t_send = self.t_hello = self.last_rx = 0.0
        self.next_id = 1

    @staticmethod
    def local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('10.255.255.255', 1))        # nessun pacchetto parte: serve solo a scegliere la scheda di rete
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return '127.0.0.1'

    def _fetch_public(self):
        try:        # chiede a ipify il tuo indirizzo pubblico (solo quando crei la partita)
            self.pub_ip = urllib.request.urlopen('https://api.ipify.org', timeout=4).read().decode().strip()
        except Exception:
            pass
        self._upnp_open()

    # ---- UPnP: apre da solo la porta UDP sul router (se lo supporta), cosi un amico su un'altra rete puo entrare
    def _upnp_open(self):
        self.upnp = 'Cerco il router (UPnP)...'
        self.upnp_ctl = None
        try:
            ssdp = ('M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: "ssdp:discover"\r\nMX: 2\r\n'
                    'ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n').encode()
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(2.5)
            s.sendto(ssdp, ('239.255.255.250', 1900))
            loc = None
            try:
                while loc is None:
                    data, _ = s.recvfrom(4096)
                    m = re.search(r'(?i)^location:\s*(\S+)', data.decode(errors='ignore'), re.M)
                    if m:
                        loc = m.group(1)
            except socket.timeout:
                pass
            s.close()
            if not loc:
                raise RuntimeError('nessun router UPnP')
            import xml.etree.ElementTree as ET
            from urllib.parse import urljoin
            xml = urllib.request.urlopen(loc, timeout=3).read().decode(errors='ignore')
            xml = re.sub(r'\sxmlns="[^"]+"', '', xml, count=1)
            root = ET.fromstring(xml)
            ctl = svc = None
            for sv in root.iter('service'):
                st = sv.findtext('serviceType') or ''
                if 'WANIPConnection' in st or 'WANPPPConnection' in st:
                    svc, ctl = st, urljoin(loc, sv.findtext('controlURL') or '')
                    break
            if not ctl:
                raise RuntimeError('servizio WAN non trovato')

            def soap(action, args):
                body = ('<?xml version="1.0"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
                        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"><s:Body><u:%s xmlns:u="%s">%s</u:%s>'
                        '</s:Body></s:Envelope>' % (action, svc, ''.join('<%s>%s</%s>' % (k, v, k) for k, v in args), action))
                rq = urllib.request.Request(ctl, body.encode(), {'Content-Type': 'text/xml; charset="utf-8"',
                                                                   'SOAPAction': '"%s#%s"' % (svc, action)})
                return urllib.request.urlopen(rq, timeout=4).read().decode(errors='ignore')

            ok = False
            for lease in ('0', '7200'):
                try:
                    soap('AddPortMapping', [('NewRemoteHost', ''), ('NewExternalPort', self.port), ('NewProtocol', 'UDP'),
                                            ('NewInternalPort', self.port), ('NewInternalClient', self.local_ip()),
                                            ('NewEnabled', 1), ('NewPortMappingDescription', 'FlightWorld'),
                                            ('NewLeaseDuration', lease)])
                    ok = True
                    break
                except Exception:
                    pass
            if not ok:
                raise RuntimeError('il router ha rifiutato la richiesta')
            self.upnp_ctl = (soap, self.port)
            ext = None
            try:
                ext = re.search(r'<NewExternalIPAddress>([^<]+)<', soap('GetExternalIPAddress', [])).group(1)
            except Exception:
                pass
            if ext and ext != '0.0.0.0':
                if self.pub_ip and ext != self.pub_ip:
                    self.upnp = ('Porta aperta, ma il tuo provider condivide l\'indirizzo (CGNAT): da Internet usa una VPN '
                                 '(Tailscale, ZeroTier, Radmin) e dai il codice con l\'IP della VPN')
                else:
                    self.pub_ip = self.pub_ip or ext
                    self.upnp = 'Porta UDP %d aperta automaticamente sul router (UPnP)' % self.port
            else:
                self.upnp = 'Porta UDP %d aperta automaticamente sul router (UPnP)' % self.port
        except Exception as ex:
            self.upnp = ('UPnP non disponibile (%s): apri a mano la porta UDP %d sul router, oppure usa una VPN '
                         '(Tailscale, ZeroTier, Radmin)' % (ex, self.port))

    def _upnp_close(self):
        c = getattr(self, 'upnp_ctl', None)
        self.upnp_ctl = None
        if c:
            try:
                c[0]('DeletePortMapping', [('NewRemoteHost', ''), ('NewExternalPort', c[1]), ('NewProtocol', 'UDP')])
            except Exception:
                pass

    def start_host(self, name, port=NET_PORT):
        self.stop()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(('', port))
            s.setblocking(False)
        except OSError as ex:
            self.status = 'Non riesco ad aprire la porta UDP %d (%s)' % (port, ex.strerror or ex)
            return False
        self.sock, self.role, self.joined, self.name = s, 'host', True, name
        self.port = port
        self.token = ''.join(random.choice(CODE_CHARS) for _ in range(5))
        self.my_id, self.next_id = 0, 1
        self.states, self.peers = {}, {}
        self.status = 'Partita aperta. Aspetto i giocatori...'
        threading.Thread(target=self._fetch_public, daemon=True).start()
        return True

    def code(self):
        return '%s:%d-%s' % (self.pub_ip or self.local_ip(), self.port, self.token)

    def local_code(self):
        return '%s:%d-%s' % (self.local_ip(), self.port, self.token)

    def start_client(self, code, name):
        self.stop()
        m = re.match(r'^\s*([\w.\-]+)(?::(\d{1,5}))?-([A-Za-z0-9]{3,8})\s*$', code)
        if not m:
            self.status = 'Codice non valido (formato: indirizzo:porta-CODICE)'
            return False
        try:
            addr = (socket.gethostbyname(m.group(1)), int(m.group(2) or NET_PORT))
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(('', 0))
            s.setblocking(False)
        except OSError as ex:
            self.status = 'Indirizzo non raggiungibile (%s)' % (ex.strerror or ex)
            return False
        self.sock, self.role, self.joined, self.name = s, 'client', False, name
        self.host_addr, self.token = addr, m.group(3).upper()
        self.states = {}
        self.last_rx = time.time()
        self.status = 'Mi collego a %s:%d ...' % addr
        return True

    def stop(self):
        if self.role == 'host':
            threading.Thread(target=self._upnp_close, daemon=True).start()
        self.upnp = ''
        if self.sock:
            try:
                if self.role == 'client' and self.joined:
                    self.sock.sendto(b'{"t":"B"}', self.host_addr)
                elif self.role == 'host':
                    for a in self.peers:
                        self.sock.sendto(b'{"t":"X"}', a)
                self.sock.close()
            except Exception:
                pass
        self.sock, self.role, self.joined = None, None, False
        self.peers, self.states = {}, {}

    def _send(self, msg, addr):
        try:
            self.sock.sendto(json.dumps(msg, separators=(',', ':')).encode(), addr)
        except Exception:
            pass

    def poll(self, now):
        if not self.sock:
            return
        for _ in range(300):
            if not self.sock:
                return
            try:
                data, addr = self.sock.recvfrom(8192)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:                       # Windows segnala qui i pacchetti respinti: si ignora
                continue
            try:
                msg = json.loads(data.decode())
                t = msg['t']
            except Exception:
                continue
            if self.role == 'host':
                self._host_msg(t, msg, addr, now)
            else:
                self._client_msg(t, msg, now)
        if self.role == 'host':
            for a in [a for a, p in self.peers.items() if now - p['last'] > 6.0]:
                self.states.pop(self.peers[a]['id'], None)
                del self.peers[a]
        elif self.role == 'client' and now - self.last_rx > 8.0:
            self.status = 'Connessione persa con l\'host'
            self.joined = False

    def _host_msg(self, t, msg, addr, now):
        if t == 'H':
            if str(msg.get('k', '')).upper() != self.token or (addr not in self.peers and len(self.peers) >= NET_MAX - 1):
                self._send({'t': 'X'}, addr)
                return
            if addr not in self.peers:
                nm = re.sub(r'[^\w .-]', '', str(msg.get('n', 'PILOTA')))[:14] or 'PILOTA'
                self.peers[addr] = dict(id=self.next_id, name=nm, last=now)
                self.next_id += 1
                self.status = '%s e entrato' % nm
            self.peers[addr]['last'] = now
            self._send({'t': 'W', 'id': self.peers[addr]['id']}, addr)
        elif addr in self.peers:
            p = self.peers[addr]
            p['last'] = now
            if t == 'S':
                st = _clean_state(msg.get('s', {}))
                if st:
                    st['id'], st['n'] = p['id'], p['name']
                    self.states[p['id']] = (st, now)
            elif t == 'B':
                self.states.pop(p['id'], None)
                del self.peers[addr]
                self.status = '%s e uscito' % p['name']

    def _client_msg(self, t, msg, now):
        self.last_rx = now
        if t == 'W':
            self.my_id = int(msg.get('id', 0))
            if not self.joined:
                self.status = 'Collegato (sono il giocatore %d)' % self.my_id
            self.joined = True
        elif t == 'X':
            self.status = 'Codice rifiutato o partita piena'
            self.stop()
        elif t == 'R' and self.joined:
            new = {}
            for raw in msg.get('p', [])[:NET_MAX]:
                st = _clean_state(raw)
                if st and st['id'] != self.my_id:
                    new[st['id']] = (st, now)
            self.states = new

    def send_state(self, st, now):
        if not self.sock or now - self.t_send < 1.0 / NET_HZ:
            return
        self.t_send = now
        if self.role == 'host':
            st = dict(st, id=0, n=self.name)
            self.states[0] = (st, now)
            for addr, p in list(self.peers.items()):
                others = [s for i, (s, t) in self.states.items() if i != p['id'] and now - t < 3.0]
                self._send({'t': 'R', 'p': others}, addr)
        elif self.role == 'client':
            if self.joined:
                self._send({'t': 'S', 's': dict(st, id=self.my_id, n=self.name)}, self.host_addr)
            elif now - self.t_hello > 1.0:
                self.t_hello = now
                self._send({'t': 'H', 'k': self.token, 'n': self.name}, self.host_addr)

    def players(self):
        names = [self.name]
        if self.role == 'host':
            names += [p['name'] for p in self.peers.values()]
        else:
            names += [s['n'] for s, t in self.states.values()]
        return names


_COL_PTS = {}


def collision_points(sp):
    """Punti del corpo (naso, coda, ali, motori e linee lungo fusoliera e ali) per il controllo degli scontri in volo."""
    if sp.short not in _COL_PTS:
        pts = [Vec3(p) for n, p in sp.points]
        ys = [p.y for p in pts]
        for y in range(int(min(ys)), int(max(ys)) + 1, 4):
            pts.append(Vec3(0, y, 0))
        tips = [p for n, p in sp.points if 'ala' in n]
        if tips:
            tip = tips[0]
            tx = max(abs(tip.x), 1.0)
            for x in range(-int(tx), int(tx) + 1, 4):
                pts.append(Vec3(x, tip.y * abs(x) / tx, tip.z))
        _COL_PTS[sp.short] = pts
    return _COL_PTS[sp.short]


def planes_collide(a, b, dist=3.2):
    """True se i due aerei (con la loro forma) si toccano."""
    if (a.pos - b.pos).length() > a.sp.cam[0] * 3.0 + b.sp.cam[0] * 3.0:
        return False
    pa = [a.pos + a.right * p.x + a.fwd * p.y + a.up * p.z for p in collision_points(a.sp)]
    pb = [b.pos + b.right * p.x + b.fwd * p.y + b.up * p.z for p in collision_points(b.sp)]
    d2 = dist * dist
    return any((x - y).lengthSquared() < d2 for x in pa for y in pb)


class RemotePlane:
    """Aereo di un altro giocatore: stesso modello e stessa livrea, mosso dai pacchetti ricevuti."""

    def __init__(self, app, st):
        self.app = app
        self.key = None
        self.ac = None
        self.label = None
        self.crashed = False
        self.t_label = 0.0
        self.build(st)

    def build(self, st):
        c = st['c']
        key = (st['s'], tuple(round(x, 2) for x in c[0]), tuple(round(x, 2) for x in c[1]), c[2], c[3], c[4])
        if key == self.key:
            return
        self.key = key
        self.destroy(keep_label=True)
        sp = SPECS[st['s']]
        old = CUSTOM.get(sp.short)
        k0 = TEX_K[0]
        CUSTOM[sp.short] = dict(old or {}, body=tuple(c[0]), accent=tuple(c[1]), name=c[2], reg=c[3], livery=c[4])
        TEX_K[0] = 1
        apply_custom(sp)
        try:
            self.ac = Aircraft(self.app.terrain, sp)
        finally:
            TEX_K[0] = k0
            if old is None:
                CUSTOM.pop(sp.short, None)
            else:
                CUSTOM[sp.short] = old
            apply_custom(sp)
        self.ac.model.reparentTo(self.app.render)
        self.ac.set_cockpit(False)
        self.ac.pos = Vec3(*st['p'])
        self.ac.fwd, self.ac.up = Vec3(*st['f']), Vec3(*st['u'])
        self.ctl = Controls()
        if self.label is None:
            tn = TextNode('nome')
            tn.setAlign(TextNode.ACenter)
            tn.setTextColor(1, 1, 0.6, 1)
            tn.setShadow(0.06, 0.06)
            tn.setShadowColor(0, 0, 0, 1)
            self.label = self.app.render.attachNewNode(tn)
            self.label.setBillboardPointEye()
            self.label.setLightOff()
            self.label.setDepthTest(False)
            self.label.setDepthWrite(False)
            self.label.setBin('fixed', 40)

    def update(self, st, age, dt):
        ac = self.ac
        self.build(st)
        ac = self.ac
        ac.model.show() if not st['hd'] else ac.model.hide()
        pred = Vec3(*st['p']) + Vec3(*st['v']) * min(age, 0.5)
        ac.pos = ac.pos + (pred - ac.pos) * min(1.0, dt * 8.0)
        k = min(1.0, dt * 8.0)
        f = ac.fwd + (Vec3(*st['f']) - ac.fwd) * k
        u = ac.up + (Vec3(*st['u']) - ac.up) * k
        f.normalize()
        u = u - f * u.dot(f)
        u.normalize()
        ac.fwd, ac.up, ac.right = f, u, f.cross(u)
        ac.vel = Vec3(*st['v'])
        ac.gear_pos, ac.throttle, ac.flaps, ac.on_ground = st['g'], st['th'], st['fl'], st['og']
        ac.eng, ac.spool = True, 1.0
        self.ctl.elev, self.ctl.ail, self.ctl.rud = st['ct']
        ac.update_visual(dt, self.ctl)
        if st['cr'] and not self.crashed:
            fx = self.app.fx
            fx.k, fx.water = clamp((ac.sp.mass / 60000.0) ** 0.34, 0.45, 2.2), False
            fx.blast(Vec3(ac.pos), fx.k, 0.9)
            ac.model.setColorScale(0.26, 0.24, 0.23, 1)
            mine = self.app.ac            # se era vicino a me, lo scontro e avvenuto anche per me: esplodo anch'io
            if mine and not mine.crashed and not mine.on_ground and (mine.pos - ac.pos).length() < 25.0 \
                    and not self.app.in_menu:
                mine.crash('collisione in volo con %s' % self.label.node().getText().split('  ')[0])
        elif not st['cr'] and self.crashed:
            ac.model.setColorScale(1, 1, 1, 1)
        self.crashed = st['cr']
        d = (ac.pos - Vec3(self.app.camera.getPos())).length()
        self.label.setPos(ac.pos + Vec3(0, 0, 10.0 + ac.sp.cam[1] * 0.4))
        self.label.setScale(clamp(d * 0.022, 2.5, 500.0))
        if st['hd']:
            self.label.hide()
        else:
            self.label.show()
        self.t_label -= dt
        if self.t_label <= 0:
            self.t_label = 0.5
            self.label.node().setText('%s  %.1f km' % (st['n'], d / 1000.0) if d > 400 else st['n'])

    def destroy(self, keep_label=False):
        if self.ac:
            self.ac.model.removeNode()
            self.ac = None
        if self.label and not keep_label:
            self.label.removeNode()
            self.label = None


# ---------------------------------------------------------------- passerelle d'imbarco e passeggeri
def _cyl(m, x, y, z0, z1, r, col, seg=16):
    for i in range(seg):
        a0, a1 = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
        p0 = (x + r * math.cos(a0), y + r * math.sin(a0))
        p1 = (x + r * math.cos(a1), y + r * math.sin(a1))
        m.quad([(p0[0], p0[1], z0), (p1[0], p1[1], z0), (p1[0], p1[1], z1), (p0[0], p0[1], z1)], col, (x, y, (z0 + z1) / 2))
        m.tri_flat((x, y, z1), (p0[0], p0[1], z1), (p1[0], p1[1], z1), col, (x, y, z0))


SEATS = {'Boeing 737-800': 160, 'Airbus A320': 150, 'Airbus A380': 520, 'Concorde': 100}


def build_pax_templates():
    """Dieci omini diversi (vestiti, carnagione, trolley a mano) da copiare per ogni passeggero."""
    shirts = [(0.8, 0.2, 0.2), (0.2, 0.4, 0.8), (0.9, 0.8, 0.2), (0.25, 0.6, 0.35), (0.9, 0.9, 0.9), (0.5, 0.3, 0.6),
              (0.95, 0.5, 0.15), (0.15, 0.15, 0.2)]
    skins = [(0.95, 0.78, 0.65), (0.8, 0.6, 0.45), (0.55, 0.38, 0.28), (0.35, 0.24, 0.18)]
    out = []
    for i in range(10):
        m = Mesh()
        sh, sk = shirts[i % 8] + (1,), skins[i % 4] + (1,)
        tr = ((0.12, 0.14, 0.20) if i % 2 else (0.32, 0.27, 0.20)) + (1,)
        m.box(-0.14, -0.02, -0.08, 0.08, 0.0, 0.8, tr)
        m.box(0.02, 0.14, -0.08, 0.08, 0.0, 0.8, tr)
        m.box(-0.2, 0.2, -0.1, 0.1, 0.8, 1.4, sh)
        m.box(-0.28, -0.2, -0.07, 0.07, 0.85, 1.35, sh)
        m.box(0.2, 0.28, -0.07, 0.07, 0.85, 1.35, sh)
        m.sphere(0, 0, 1.52, 0.11, 0.11, 0.13, sk, 8, 5)
        if i % 3 == 0:
            m.box(0.30, 0.48, -0.12, 0.12, 0.2, 0.75, (0.12, 0.12, 0.14, 1))
        n = NodePath(m.node('pax%d' % i))
        n.setTwoSided(True)
        out.append(n)
    return out


def build_cargo_templates():
    """Pacchi e casse di varie dimensioni e colori da far scorrere sul nastro del dock cargo."""
    cols = [(0.72, 0.52, 0.30), (0.62, 0.43, 0.24), (0.85, 0.70, 0.45), (0.30, 0.10, 0.55), (0.95, 0.45, 0.05),
            (1.0, 0.80, 0.05), (0.15, 0.55, 0.80), (0.80, 0.80, 0.82)]
    out = []
    for i in range(8):
        m = Mesh()
        w, d, h = 0.45 + 0.1 * (i % 3), 0.45 + 0.1 * ((i + 1) % 3), 0.35 + 0.12 * (i % 4)
        m.box(-w / 2, w / 2, -d / 2, d / 2, 0.0, h, cols[i] + (1,))
        m.box(-0.04, 0.04, -d / 2 - 0.005, d / 2 + 0.005, h - 0.01, h + 0.005, (0.85, 0.82, 0.7, 1))      # nastro adesivo
        if i % 3 == 0:
            m.box(-w / 4, w / 4, -d / 4, d / 4, h, h + 0.3, cols[(i + 3) % 8] + (1,))
        n = NodePath(m.node('pacco%d' % i))
        n.setTwoSided(True)
        out.append(n)
    return out


class Bridge:
    """Passerella telescopica: la punta segue la porta dell'aereo parcheggiato e si attacca da sola.
    Coordinate locali dell'aeroporto (x = u verso il terminal, y = v lungo la pista).
    kind='cargo': nastro trasportatore arancio che serve il portellone dei cargo e porta pacchi invece di persone."""
    REST = 9.0

    def __init__(self, apt, root, v, tpl, kind='pax'):
        self.apt, self.root, self.tpl, self.kind = apt, root, tpl, kind
        self.mode = 'off'
        self.left = 0.0
        self.P = Vec3(GATE_U + 1.5, v, 3.7)
        self.rest = Vec3(GATE_U - self.REST, v, 3.7)
        self.tip = Vec3(self.rest)
        self.docked = False
        self.moving = False
        self.pax = []
        self.spawn_t = 0.0
        self.boarded = 0.0
        self.cap = 0
        white, grey = (0.88, 0.89, 0.92, 1), (0.35, 0.36, 0.40, 1)
        if kind == 'cargo':
            white = (0.95, 0.55, 0.10, 1)
        m = Mesh()                                          # rotonda e colonna sul bordo del terminal
        _cyl(m, self.P.x, v, 2.2, 5.3, 2.3, white, 18)
        _cyl(m, self.P.x, v, 0.0, 2.2, 0.55, grey, 10)
        m.box(self.P.x - 2.3, self.P.x + 0.2, v - 0.5, v + 0.5, 5.3, 5.5, (0.80, 0.12, 0.12, 1))
        root.attachNewNode(m.node('rotonda'))
        t = Mesh()                                          # tubo: lungo 1 lungo +y, si scala in lunghezza
        t.box(-1.3, 1.3, 0, 1, -1.25, -1.10, grey)
        t.box(-1.4, 1.4, 0, 1, 1.05, 1.25, white)
        for s in (-1, 1):
            t.box(s * 1.30 if s > 0 else -1.40, 1.40 if s > 0 else -1.30, 0, 1, -1.25, -0.80, white)
        self.tube = root.attachNewNode(t.node('tubo'))
        gm = Mesh()
        for s in (-1, 1):
            gm.quad([(s * 1.36, 0, -0.80), (s * 1.36, 1, -0.80), (s * 1.36, 1, 1.05), (s * 1.36, 0, 1.05)],
                    (0.55, 0.75, 0.90, 0.32), (0, 0.5, 0))
        self.glass = root.attachNewNode(gm.node('vetro'))
        self.glass.setTransparency(TransparencyAttrib.MAlpha)
        self.glass.setTwoSided(True)
        self.glass.setDepthWrite(False)
        cm = Mesh()
        cm.box(-1.65, 1.65, -3.2, 0.0, -1.35, 1.35, (0.82, 0.84, 0.88, 1))
        cm.box(-1.67, 1.67, -3.2, 0.0, 0.55, 0.80, (0.10, 0.30, 0.60, 1) if kind == 'pax' else (0.15, 0.15, 0.17, 1))
        cm.box(-1.40, 1.40, 0.0, 0.7, -1.20, 1.20, (0.06, 0.06, 0.07, 1))
        self.cab = root.attachNewNode(cm.node('cabina'))
        colm = Mesh()
        colm.box(-0.35, 0.35, -0.35, 0.35, 0.0, 1.0, grey)
        colm.box(-1.0, 1.0, -0.25, 0.25, 0.0, 0.35, (0.1, 0.1, 0.12, 1))
        self.col = root.attachNewNode(colm.node('colonna'))
        self.gl = make_points([(0.0, 0.0, 0.0, 1.0, 0.85, 0.5, 1.0), (0.0, 0.0, 0.5, 1.0, 0.85, 0.5, 1.0)], 9.0, root, 'luce_cabina')
        GLOW.append((self.gl, 'steady'))
        self.apply()

    def apply(self):
        d = self.tip - self.P
        ln = max(d.length(), 0.5)
        self.tube.setPos(self.P)
        self.tube.lookAt(Point3(self.tip))
        self.tube.setScale(1, ln, 1)
        self.glass.setPos(self.P)
        self.glass.setHpr(self.tube.getHpr())
        self.glass.setScale(1, ln, 1)
        self.cab.setPos(self.tip)
        self.cab.setHpr(self.tube.getHpr())
        self.gl.setPos(self.tip + (self.P - self.tip) * (0.5 / max(ln, 0.5)))
        c = self.P + d * 0.8
        self.col.setPos(c.x, c.y, 0.0)
        self.col.setScale(1, 1, max(0.5, c.z - 1.3))
        self.col.setH(self.tube.getH())

    def update(self, dt, target, cap, ac=None):
        goal = target if target is not None else self.rest
        d = goal - self.tip
        ln = d.length()
        self.moving = ln > 0.02
        if self.moving:
            self.tip = self.tip + d * min(1.0, 3.5 * dt / ln)
            self.apply()
        self.docked = target is not None and (goal - self.tip).length() < 0.15
        if target is None:
            if self.mode == 'in' and ac is not None:
                ac.occ = int(self.boarded)           # chi e salito resta a bordo (si scarichera al prossimo attracco)
            self.mode = 'off'
            self.boarded = 0.0
            self.clear_pax()
            return
        self.cap = cap
        if self.docked:
            pk = self.kind == 'cargo'
            if self.mode == 'off':                   # appena attaccato: se c'e gente/merce a bordo si scarica prima
                occ = getattr(ac, 'occ', 0) if ac is not None else 0
                self.left = float(occ)
                self.mode = 'out' if occ > 0 else 'in'
                self.boarded = 0.0
            if self.mode == 'out':
                step = max(cap, 1) / 30.0 * dt
                self.left = max(0.0, self.left - step)
                if ac is not None:
                    ac.occ = int(math.ceil(self.left))
                self.spawn_t -= dt
                if self.spawn_t <= 0.0 and len(self.pax) < 16 and self.left > 1.0:
                    self.spawn_t = random.uniform(0.25, 0.55)
                    n = random.choice(self.tpl).copyTo(self.root)
                    self.pax.append(dict(n=n, t=1.0, ph=random.uniform(0, 6.3), lat=random.uniform(-0.8, 0.8),
                                         sp=random.uniform(1.4, 2.0), d=-1))
                if self.left <= 0.0 and not self.pax:
                    self.mode = 'in'
                    self.boarded = 0.0
            elif self.boarded < cap:
                self.boarded = min(cap, self.boarded + cap / 35.0 * dt)
                self.spawn_t -= dt
                if self.spawn_t <= 0.0 and len(self.pax) < 16:
                    self.spawn_t = random.uniform(0.25, 0.55)
                    n = random.choice(self.tpl).copyTo(self.root)
                    self.pax.append(dict(n=n, t=0.0, ph=random.uniform(0, 6.3), lat=random.uniform(-0.8, 0.8),
                                         sp=random.uniform(1.4, 2.0), d=1))
            tube = self.tip - self.P
            ln2 = max(tube.length(), 0.5)
            dirv = tube / ln2
            keep = []
            for p in self.pax:
                dr = p.get('d', 1)
                p['t'] += dr * p['sp'] * dt / ln2
                if p['t'] >= 1.0 or p['t'] <= 0.0:
                    p['n'].removeNode()
                    continue
                p['ph'] += dt * 9.0
                pos = self.P + tube * p['t'] + Vec3(-dirv.y, dirv.x, 0) * p['lat'] * 0.8
                bob = 0.0 if self.kind == 'cargo' else 0.035 * abs(math.sin(p['ph']))
                p['n'].setPos(pos.x, pos.y, self.P.z + tube.z * p['t'] - 1.10 + bob)
                p['n'].setH(-math.degrees(math.atan2(dirv.x, dirv.y)) + (180.0 if dr < 0 else 0.0) +
                            (0.0 if self.kind == 'cargo' else 3.0 * math.sin(p['ph'] * 0.5)))
                keep.append(p)
            self.pax = keep

    def clear_pax(self):
        for p in self.pax:
            p['n'].removeNode()
        self.pax = []


# ---------------------------------------------------------------- schermata di caricamento animata
def _logo_image(size=1024):
    """Logo FlightWorld: disco blu con anello e aereo bianco (stesso disegno che compare alla fine dell'animazione)."""
    S = size
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    cx = cy = S / 2.0
    d = np.hypot(xx - cx, yy - cy) / (S * 0.44)
    hl = np.hypot(xx - S * 0.38, yy - S * 0.34) / (S * 0.6)
    t = np.clip(0.25 + 0.75 * hl, 0, 1)
    img = np.zeros((S, S, 4), np.float32)
    img[..., 0] = 0.36 * (1 - t) + 0.03 * t
    img[..., 1] = 0.68 * (1 - t) + 0.20 * t
    img[..., 2] = 1.00 * (1 - t) + 0.62 * t
    img[..., 3] = np.clip((1.0 - d) * S * 0.44, 0, 1)
    im = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
    dr = ImageDraw.Draw(im)
    R = S * 0.455
    dr.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(255, 255, 255, 235), width=int(S * 0.012))
    R2 = S * 0.37
    dr.arc([cx - R2, cy - R2 * 0.42, cx + R2, cy + R2 * 0.42], 200, 340, fill=(255, 255, 255, 120), width=int(S * 0.006))
    dr.arc([cx - R2, cy - R2 * 0.42, cx + R2, cy + R2 * 0.42], 20, 160, fill=(255, 255, 255, 70), width=int(S * 0.006))
    a = math.radians(40.0)
    ca, sa = math.cos(a), math.sin(a)
    sc = S * 0.27

    def P(pts):
        return [(cx + sc * (x * ca - y * sa), cy + sc * (x * sa + y * ca)) for x, y in pts]
    dr.polygon(P([(0, -1.0), (0.09, -0.8), (0.09, 0.5), (0.04, 0.95), (-0.04, 0.95), (-0.09, 0.5), (-0.09, -0.8)]), fill=(255, 255, 255, 255))
    dr.polygon(P([(0.09, -0.15), (0.95, 0.38), (0.95, 0.52), (0.09, 0.30), (-0.09, 0.30), (-0.95, 0.52), (-0.95, 0.38), (-0.09, -0.15)]),
               fill=(255, 255, 255, 255))
    dr.polygon(P([(0.05, 0.72), (0.40, 0.98), (0.40, 1.08), (0.05, 0.95), (-0.05, 0.95), (-0.40, 1.08), (-0.40, 0.98), (-0.05, 0.72)]),
               fill=(255, 255, 255, 255))
    return im


def _glow_image(size=256):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    a = np.clip(1.0 - d, 0, 1) ** 2.2
    img = np.zeros((size, size, 4), np.float32)
    img[..., 0], img[..., 1], img[..., 2], img[..., 3] = 0.35, 0.65, 1.0, a
    return Image.fromarray((img * 255).astype(np.uint8), 'RGBA')


def _earth_land(x, y, z):
    return (math.sin(2.3 * x + 1.0) * math.cos(2.1 * y - 0.5) + 0.8 * math.sin(3.7 * z + 2.0) * math.cos(3.1 * x + 0.7) +
            0.5 * math.sin(5.3 * y + 3.0) * math.sin(4.7 * z - 1.0) + 0.35 * math.sin(9.0 * x + 4.0 * y) * math.sin(8.0 * z))


def _earth_image(W=1024, H=512):
    la = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * math.pi
    lo = ((np.arange(W, dtype=np.float32) + 0.5) / W) * 2 * math.pi
    LO, LA = np.meshgrid(lo, la)
    x, y, z = np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)
    v = (np.sin(2.3 * x + 1.0) * np.cos(2.1 * y - 0.5) + 0.8 * np.sin(3.7 * z + 2.0) * np.cos(3.1 * x + 0.7) +
         0.5 * np.sin(5.3 * y + 3.0) * np.sin(4.7 * z - 1.0) + 0.35 * np.sin(9.0 * x + 4.0 * y) * np.sin(8.0 * z) +
         0.22 * np.sin(17.0 * x + 6.0 * z) * np.sin(15.0 * y - 2.0) + 0.14 * np.sin(31.0 * y + 11.0 * x) * np.sin(27.0 * z + 1.0))
    ss = lambda a, b, q: np.clip((q - a) / (b - a), 0, 1) ** 2 * (3 - 2 * np.clip((q - a) / (b - a), 0, 1))
    land = ss(0.40, 0.41, v)[..., None]              # stile piatto: solo mare e terra, bordi netti
    col = np.array([0.10, 0.42, 0.95], np.float32) * (1 - land) + np.array([0.40, 0.80, 0.48], np.float32) * land
    im = np.concatenate([col, np.ones((H, W, 1), np.float32)], -1)
    return Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8), 'RGBA')


def _tex_sphere(nlon=96, nlat=48):
    fmt = GeomVertexFormat.getV3n3t2()
    vd = GeomVertexData('terra', fmt, Geom.UHStatic)
    vw, nw, tw = GeomVertexWriter(vd, 'vertex'), GeomVertexWriter(vd, 'normal'), GeomVertexWriter(vd, 'texcoord')
    for la in range(nlat + 1):
        th = math.pi * la / nlat
        rr, z = math.sin(th), math.cos(th)
        for lo in range(nlon + 1):
            ph = 2 * math.pi * lo / nlon
            x, y = rr * math.cos(ph), rr * math.sin(ph)
            vw.addData3(x, y, z)
            nw.addData3(x, y, z)
            tw.addData2(lo / float(nlon), la / float(nlat))
    prim = GeomTriangles(Geom.UHStatic)
    for la in range(nlat):
        for lo in range(nlon):
            a = la * (nlon + 1) + lo
            b, c, d = a + 1, a + nlon + 2, a + nlon + 1
            prim.addVertices(a, c, b)
            prim.addVertices(a, d, c)
    g = Geom(vd)
    g.addPrimitive(prim)
    gn = GeomNode('terra')
    gn.addGeom(g)
    return gn


def _plane_mesh():
    """Piccolo aereo di linea (muso verso +Y) per l'animazione di caricamento."""
    m = Mesh()
    W, G, A = (0.96, 0.97, 1.0, 1), (0.72, 0.76, 0.84, 1), (0.95, 0.70, 0.12, 1)
    m.loft([(-3.1, 0.04, 0.04, 0.2), (-2.6, 0.26, 0.28, 0.1), (-1.0, 0.36, 0.38, 0.0), (1.8, 0.36, 0.38, 0.0),
            (2.7, 0.24, 0.26, 0.0), (3.3, 0.02, 0.02, 0.0)], 14, W)
    for s in (-1, 1):
        m.hull([(s * 0.3, 0.6, -0.04), (s * 3.6, -0.7, -0.04), (s * 3.6, -1.25, -0.04), (s * 0.3, -0.5, -0.04),
                (s * 0.3, 0.6, 0.04), (s * 3.6, -0.7, 0.04), (s * 3.6, -1.25, 0.04), (s * 0.3, -0.5, 0.04)], G)
        m.hull([(s * 0.2, -2.55, 0.0), (s * 1.3, -3.15, 0.0), (s * 1.3, -3.4, 0.0), (s * 0.2, -2.95, 0.0),
                (s * 0.2, -2.55, 0.05), (s * 1.3, -3.15, 0.05), (s * 1.3, -3.4, 0.05), (s * 0.2, -2.95, 0.05)], G)
    m.hull([(-0.03, -2.2, 0.3), (0.03, -2.2, 0.3), (0.03, -3.4, 0.3), (-0.03, -3.4, 0.3),
            (-0.03, -2.9, 1.5), (0.03, -2.9, 1.5), (0.03, -3.5, 1.5), (-0.03, -3.5, 1.5)], A)
    return m


_T0 = time.time()


class LoadingIntro:
    """Terra che gira con un aereo in orbita: l'aereo atterra sul pianeta, la Terra diventa blu e si trasforma nel logo."""
    T_LAND, T_BLUE, T_MORPH, T_END = 5.4, 6.8, 8.4, 9.8

    def __init__(self, base):
        self.b = base
        self.t0 = time.time()
        base.render2d.hide()
        self.last = 0.0
        self.prog = 0.0
        self.shown = 0.0
        self.alive = True
        self.sc = NodePath('ldscene')
        cam = Camera('ldcam')
        lens = PerspectiveLens()
        lens.setFov(38)
        lens.setNearFar(0.5, 400.0)
        w, h = base.win.getXSize(), base.win.getYSize()
        lens.setAspectRatio(float(w) / max(1, h))
        cam.setLens(lens)
        self.cam = self.sc.attachNewNode(cam)
        self.dr = base.win.makeDisplayRegion()
        self.dr.setSort(5)
        self.dr.setCamera(self.cam)
        self.dr.setClearColorActive(True)
        self.dr.setClearColor((0.008, 0.016, 0.04, 1))
        self.dr.setClearDepthActive(True)
        al = AmbientLight('ldamb')
        al.setColor((0.32, 0.36, 0.46, 1))
        self.sc.setLight(self.sc.attachNewNode(al))
        dl = DirectionalLight('ldsun')
        dl.setColor((1.0, 0.96, 0.88, 1))
        dln = self.sc.attachNewNode(dl)
        dln.setHpr(35, -25, 0)
        self.sc.setLight(dln)
        self.C = Vec3(0, 24.0, 0)
        self.r = 2.5
        # stelle
        ls = LineSegs()
        ls.setThickness(2.0)
        rnd = random.Random(5)
        for _ in range(70):
            a, e = rnd.uniform(0, 6.283), rnd.uniform(-0.9, 0.9)
            d = Vec3(math.cos(a) * math.cos(e), abs(math.sin(a)) * math.cos(e) + 0.35, math.sin(e)) * 150.0
            c = rnd.uniform(0.45, 1.0)
            ls.setColor(c, c, min(1.0, c + 0.15), 1)
            ls.moveTo(d)
            ls.drawTo(d + Vec3(0.25, 0, 0.0))
        self.stars = self.sc.attachNewNode(ls.create())
        self.stars.setLightOff()
        self.stars.setPos(0, 0, 0)
        # alone
        cm = CardMaker('glow')
        cm.setFrame(-1, 1, -1, 1)
        self.glow = self.sc.attachNewNode(cm.generate())
        gt = _make_tex(_glow_image(), 'ldglow')
        gt.setWrapU(Texture.WM_clamp)
        self.glow.setTexture(gt)
        self.glow.setTransparency(TransparencyAttrib.MAlpha)
        self.glow.setLightOff()
        self.glow.setDepthWrite(False)
        self.glow.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha, ColorBlendAttrib.OOne))
        self.glow.setPos(self.C + Vec3(0, 1.0, 0))
        self.glow.setScale(self.r * 1.55)
        # terra
        self.flat = self.sc.attachNewNode('flat')
        self.flat.setPos(self.C)
        self.tilt = self.flat.attachNewNode('tilt')
        self.tilt.setP(18)
        self.tilt.setR(-12)
        self.spin = self.tilt.attachNewNode('spin')
        self.earth = self.spin.attachNewNode(_tex_sphere())
        self.earth.setTexture(_make_tex(_earth_image(), 'ldterra'))
        self.earth.setLightOff()
        self.earth.setScale(self.r)
        self.sea = self.flat.attachNewNode(self._sphere_mesh(1.0, False).node('mare'))
        self.sea.setScale(self.r * 1.006)
        self.sea.setTransparency(TransparencyAttrib.MAlpha)
        self.sea.setLightOff()
        self.sea.setColorScale(1, 1, 1, 0)
        # aereo
        pm = _plane_mesh()
        self.plane = self.sc.attachNewNode(pm.node('aereo'))
        self.plane.setTransparency(TransparencyAttrib.MAlpha)
        self.plane.setTwoSided(True)
        self.plane.setLightOff()
        self.plane.setColor(1, 1, 1, 1, 1)
        self.trail = None
        # logo
        cm = CardMaker('logo')
        cm.setFrame(-1, 1, -1, 1)
        self.logo = self.sc.attachNewNode(cm.generate())
        lt = _make_tex(_logo_image(), 'ldlogo')
        lt.setWrapU(Texture.WM_clamp)
        lt.setWrapV(Texture.WM_clamp)
        self.logo.setTexture(lt)
        self.logo.setTransparency(TransparencyAttrib.MAlpha)
        self.logo.setLightOff()
        self.logo.setPos(self.C + Vec3(0, -self.r * 0.06, 0))
        self.logo.setScale(self.r * 1.138)
        self.logo.setColorScale(1, 1, 1, 0)
        # livello 2D proprio (sopra tutto, cosi non si vedono HUD o menu che stanno nascendo)
        self.s2 = NodePath('ld2d')
        c2 = Camera('ld2cam')
        l2 = OrthographicLens()
        asp = float(w) / max(1, h)
        l2.setFilmSize(2.0 * asp, 2.0)
        l2.setNearFar(-20, 20)
        c2.setLens(l2)
        self.cam2 = self.s2.attachNewNode(c2)
        self.cam2.setPos(0, -5, 0)
        self.dr2 = base.win.makeDisplayRegion()
        self.dr2.setSort(30)
        self.dr2.setCamera(self.cam2)
        self.dr2.setClearDepthActive(True)
        self.s2.setDepthTest(False)
        self.s2.setDepthWrite(False)
        # testi e barra
        self.title = OnscreenText(parent=self.s2, text='FLIGHTWORLD', pos=(0, -0.80), scale=0.12, fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.7))
        self.title.setTransparency(TransparencyAttrib.MAlpha)
        self.sub = OnscreenText(parent=self.s2, text='SIMULATORE DI VOLO', pos=(0, -0.88), scale=0.04, fg=(1, 0.78, 0.25, 1))
        self.sub.setTransparency(TransparencyAttrib.MAlpha)
        self.msg = OnscreenText(parent=self.s2, text='', pos=(-0.62, -0.955), scale=0.034, fg=(0.7, 0.8, 0.95, 1), align=TextNode.ALeft, mayChange=True)
        self.pct = OnscreenText(parent=self.s2, text='', pos=(0.62, -0.955), scale=0.034, fg=(1, 0.78, 0.25, 1), align=TextNode.ARight, mayChange=True)
        self.track = DirectFrame(parent=self.s2, frameColor=(1, 1, 1, 0.12), frameSize=(-0.62, 0.62, -0.918, -0.906))
        self.fill = DirectFrame(parent=self.s2, frameColor=(1, 0.78, 0.25, 1), frameSize=(-0.62, -0.62, -0.918, -0.906))
        self.set_time(0.0)

    def _sphere_mesh(self, rad, land, nlon=72, nlat=36):
        m = Mesh()
        rings = []
        for la in range(nlat + 1):
            th = math.pi * la / nlat
            rr, z = math.sin(th), math.cos(th)
            ring = []
            for lo in range(nlon + 1):
                ph = 2 * math.pi * lo / nlon
                x, y = rr * math.cos(ph), rr * math.sin(ph)
                if land:
                    v = _earth_land(x, y, z)
                    sh = 0.5 + 0.5 * math.sin(7.0 * x + 3.0 * y + 2.0 * z)
                    if abs(z) > 0.90 + 0.03 * math.sin(9 * x + 5 * y):
                        col = (0.93, 0.96, 1.0, 1)
                    elif v > 0.42:
                        col = (0.20 + 0.25 * sh, 0.50 + 0.12 * sh, 0.20, 1) if v < 1.05 else (0.55, 0.45, 0.28, 1)
                    elif v > 0.32:
                        col = (0.35, 0.62, 0.75, 1)               # acque basse
                    else:
                        col = (0.04 + 0.03 * sh, 0.20 + 0.05 * sh, 0.52 + 0.08 * sh, 1)
                else:
                    col = (0.10, 0.42, 0.98, 1)
                ring.append(m.vert((x * rad, y * rad, z * rad), (x, y, z), col))
            rings.append(ring)
        for k in range(nlat):
            for s in range(nlon):
                a, b = rings[k][s], rings[k][s + 1]
                c, d = rings[k + 1][s + 1], rings[k + 1][s]
                m._oriented(a, b, c)
                m._oriented(a, c, d)
        return m

    # ---- animazione: tutto in funzione del tempo t
    def _orbit_pos(self, t):
        C, r = self.C, self.r
        p = clamp(t / self.T_LAND, 0.0, 1.0)
        a_end = 1.5 * math.pi + 4 * math.pi
        a = a_end - 5.2 * math.pi * (1.0 - p) ** 1.35
        R = r * (1.02 + 0.55 * (1.0 - smoothstep(0.30, 1.0, p)))
        tl = math.radians(26.0)
        ex, ey = Vec3(1, 0, 0), Vec3(0, math.cos(tl), math.sin(tl))
        return C + (ex * math.cos(a) + ey * math.sin(a)) * R, R

    def set_time(self, t):
        C, r = self.C, self.r
        ease = smoothstep(0.0, 1.3, t)
        self.spin.setH(math.degrees(0.42 * min(t, self.T_BLUE)) + 20.0 * smoothstep(self.T_BLUE, self.T_MORPH, t))
        self.stars.setR(t * 1.2)
        # aereo
        if t < self.T_LAND + 0.02:
            pos, R = self._orbit_pos(t)
            self._land_pos = pos
            p0, _ = self._orbit_pos(min(t, self.T_LAND - 0.04) - 0.03)
            p1, _ = self._orbit_pos(min(t, self.T_LAND - 0.04) + 0.03)
            tang = p1 - p0
            if tang.length() > 1e-5:
                tang.normalize()
                up = pos - C
                up.normalize()
                self.plane.setPos(pos)
                self.plane.lookAt(Point3(pos + tang), up)
                self._land_hpr = self.plane.getHpr()
            k = (R - r) / (0.57 * r)
            self.plane.setScale(0.085 + 0.06 * clamp(k, 0.0, 1.0))
            self.plane.setColorScale(1, 1, 1, 1)
            self.plane.show()
        else:
            # dopo l'atterraggio l'aereo scivola al centro del disco e diventa il disegno del logo
            q = smoothstep(self.T_BLUE, self.T_MORPH, t)
            tgt = C + Vec3(0, -r * 1.07, 0)
            pos = self._land_pos * (1 - q) + tgt * q
            self.plane.setPos(pos)
            self.plane.setHpr(self._land_hpr)
            h0 = self.plane.getHpr()
            fwd = Vec3(math.sin(math.radians(40)), 0, math.cos(math.radians(40)))
            self.plane.lookAt(Point3(pos + fwd), Vec3(0, -1, 0))
            h1 = self.plane.getHpr()
            k = smoothstep(0.0, 0.8, q)
            self.plane.setHpr(*[h0[i] + ((h1[i] - h0[i] + 180.0) % 360.0 - 180.0) * k for i in range(3)])
            self.plane.setScale(0.085 * (1.0 - 0.25 * q))
            self.plane.setColorScale(1, 1, 1, 1.0 - smoothstep(self.T_MORPH - 0.9, self.T_MORPH, t))
        # Terra -> blu -> disco -> logo
        blue = smoothstep(self.T_LAND, self.T_BLUE, t)
        self.sea.setColorScale(1, 1, 1, blue)
        fl = smoothstep(self.T_BLUE + 0.1, self.T_MORPH, t)
        sx = 0.55 + 0.45 * ease
        self.flat.setScale(sx, sx * (1.0 - 0.97 * fl), sx)
        lg = smoothstep(self.T_MORPH - 0.9, self.T_MORPH + 0.4, t)
        self.logo.setColorScale(1, 1, 1, lg)
        pulse = 0.85 + 0.15 * math.sin(t * 3.0)
        self.glow.setColorScale(0.6, 0.8, 1.0, 0.0)
        self.glow.setScale(r * (1.55 + 0.35 * fl))
        # titolo
        ta = smoothstep(self.T_MORPH - 0.4, self.T_MORPH + 0.8, t)
        self.title.setColorScale(1, 1, 1, ta)
        self.title.setPos(0, -0.80 - 0.06 * (1 - ta))
        self.sub.setColorScale(1, 1, 1, smoothstep(self.T_MORPH + 0.2, self.T_MORPH + 1.0, t))
        # barra
        self.shown += (self.prog - self.shown) * 0.25
        w = 1.24 * clamp(self.shown, 0.0, 1.0)
        self.fill['frameSize'] = (-0.62, -0.62 + w, -0.918, -0.906)
        self.pct.setText('%d%%' % int(self.shown * 100.0 + 0.5))
        # inquadratura: leggero dolly
        self.cam.setPos(0, -0.8 * smoothstep(self.T_BLUE, self.T_END, t) * 0.0, 0)

    def pump(self, prog=None, msg=None, force=False):
        if not self.alive:
            return
        if prog is not None:
            self.prog = max(self.prog, prog)
        if msg:
            self.msg.setText(msg)
        now = time.time() - self.t0
        if not force and now - self.last < 0.03:
            return
        self.last = now
        self.set_time(now)
        self.b.graphicsEngine.renderFrame()
        if os.environ.get('FW_LOADSHOT') and int(now) != getattr(self, '_shot_s', -1):
            self._shot_s = int(now)
            self.b.win.saveScreenshot(Filename.fromOsSpecific(os.path.join(os.environ['FW_LOADSHOT'], 'ld_%02d.png' % int(now))))

    def finish(self):
        """Fine del caricamento: lascia completare l'animazione, poi dissolve verso il menu."""
        self.prog = 1.0
        self.msg.setText('Pronto al decollo')
        while time.time() - self.t0 < self.T_END or self.shown < 0.995:
            self.pump(force=True)
            time.sleep(0.008)
        self.destroy()

    def destroy(self):
        if not self.alive:
            return
        self.alive = False
        self.b.render2d.show()
        self.b.win.removeDisplayRegion(self.dr)
        self.b.win.removeDisplayRegion(self.dr2)
        for x in (self.title, self.sub, self.msg, self.pct, self.track, self.fill):
            x.destroy()
        self.sc.removeNode()
        self.s2.removeNode()


class FlightSim(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.setBackgroundColor(*HORIZON)
        self.camLens.setNearFar(0.5, 70000.0)
        self.ld = LoadingIntro(self) if not ARGS.selftest else None
        _LD[0] = self.ld
        if self.ld:
            self.ld.pump(0.0, 'Accendo i motori...', True)

        self._ui_home, self._ui_iv, self._hv_iv, self._hv_home = {}, {}, {}, {}
        self.ctl = Controls()
        self.cam_mode = 0
        self.paused = False
        self.acc = 0.0
        self.hud_np = None
        self.map_np = None
        self.sim_time = 0.0
        self.mouse = (0.0, 0.0)
        self.alt_hold = False
        self.alt_target = 0.0
        self.ap_i = 0.0
        self.in_menu = False
        self.in_hangar = False
        self.h_cam = 0.9
        self.h_zoom = 1.0
        self.beacon_np = None
        self.prev_vz = 0.0
        self.was_ground = True
        self.fake_mouse = (0.0, 0.0)
        self.rud_key = 0
        self.terrain_alert = False
        self.alt_i = 0.0
        self.sel_spec = ARGS.aircraft if ARGS.selftest else 0
        self.sel_apt = ARGS.airport if ARGS.selftest else 0
        self.dest = 1
        self.reg = ButtonRegistry.ptr()
        self.fx = Effects(self)
        self.audio = Audio(self)
        self.net = Net()
        self.remotes = {}
        self.net.name = re.sub(r'[^\w .-]', '', os.environ.get('USERNAME', 'PILOTA'))[:12].upper() or 'PILOTA'

        self._ldp(0.02, 'Preparo il cielo...')
        self.setup_lights()
        self.terrain = Terrain()
        self._ldp(0.02, 'Modello il terreno...')
        self.build_terrain()
        self._ldp(0.54, 'Riempio gli oceani...')
        self.build_water()
        self.build_sky()
        self._ldp(0.58, 'Costruisco gli aeroporti...')
        self.build_airports()
        self._ldp(0.72, 'Pianto alberi e costruisco citta...')
        self.build_trees_and_houses()
        self._ldp(0.88, 'Accendo le nuvole...')
        self.build_clouds()
        self.build_minimap()
        self.ac = None
        self._ldp(0.93, 'Strumenti di bordo...')
        self.build_hud()
        self.build_menu()
        self.build_hangar()
        self.build_hangar_ui()
        self.ui_hover(self.hui)
        DirectFrame(parent=self.hui, frameColor=(1, 0.78, 0.25, 1), frameSize=(1.144, 1.15, -0.97, 0.97))
        self.build_mp_ui()
        self.build_fs_button()
        self._ldp(0.97, 'Rullo verso la pista...')
        self.new_aircraft(self.sel_spec)
        self.ac.reset(ARGS.air, APTS[self.sel_apt])
        if self.ld:
            self.ld.finish()
            self.ld = None
            _LD[0] = None
        self.cam_f = Vec3(self.ac.fwd)
        self.cam_u = Vec3(0, 0, 1)
        # visuale libera: clic destro + trascinamento, rotellina = distanza
        self.look = False
        self.look_ref = (0.0, 0.0)
        self.lk_ty = self.lk_tp = 0.0          # obiettivo (gradi)
        self.lk_y = self.lk_p = 0.0            # valore smussato
        self.cam_zoom = 1.0
        self.accept('control-b', self.toggle_fullscreen)
        self.accept('mouse3', self.look_down)
        self.accept('mouse3-up', self.look_up)
        self.accept('wheel_up', self.wheel, [-1])
        self.accept('wheel_down', self.wheel, [1])

        for key, fn in (('c', self.next_camera), ('g', self.toggle_gear), ('f', self.cycle_flaps),
                        ('r', self.toggle_althold), ('t', self.next_dest), ('n', self.restart_air),
                        ('e', self.toggle_engines), ('b', self.restart_gate), ('k', self.skip_hour),
                        ('backspace', self.restart_ground), ('tab', self.toggle_pause),
                        ('h', self.toggle_help), ('escape', self.toggle_menu)):
            self.accept(key, fn)
            self.accept('shift-' + key, fn)
        if not ARGS.selftest:
            self.show_menu(True)
        self.taskMgr.add(self.update, 'update')
        if os.environ.get('FW_LOADSHOT'):
            for k in range(90):
                self.taskMgr.step()
                if k in (4, 20, 89):
                    self.win.saveScreenshot(Filename.fromOsSpecific(os.path.join(os.environ['FW_LOADSHOT'], 'menu_%02d.png' % k)))
            sys.exit(0)

    def _ldp(self, prog=None, msg=None):
        """Aggiorna la schermata di caricamento (se attiva); chiamabile spesso dai cicli di costruzione."""
        if os.environ.get('FW_T') and prog is not None:
            print('T', round(time.time() - _T0, 2), msg)
        if getattr(self, 'ld', None):
            self.ld.pump(prog, msg, prog is not None)

    # ---- scena
    def setup_lights(self):
        amb = AmbientLight('amb')
        amb.setColor((0.50, 0.52, 0.58, 1))
        self.amb_np = self.render.attachNewNode(amb)
        self.render.setLight(self.amb_np)
        sun = DirectionalLight('sun')
        sun.setColor((1.0, 0.96, 0.88, 1))
        self.sun_np = self.render.attachNewNode(sun)
        self.sun_np.setHpr(30, -42, 0)
        self.render.setLight(self.sun_np)
        moon = DirectionalLight('luna')
        moon.setColor((0, 0, 0, 1))
        self.moon_np = self.render.attachNewNode(moon)
        self.render.setLight(self.moon_np)
        self.sun_dir = -self.render.getRelativeVector(self.sun_np, Vec3(0, 1, 0))
        self.fog = None
        self.tod = (ARGS.hour if ARGS.hour >= 0 else 8.0) % 24.0       # ora del giorno: 24 ore = 24 minuti reali
        self.night = 0.0
        self.shadow_k = 1.0
        self.env_t = 99.0
        fog = Fog('nebbia')
        fog.setColor(*HORIZON)
        fog.setExpDensity(0.000045)
        self.render.setFog(fog)
        self.fog = fog

    def build_terrain(self):
        n = GRID_N + 1
        H = self.terrain.H
        m = Mesh()
        for j in range(n):
            if j % 4 == 0:
                ld_pump(0.42 + 0.08 * j / n)
            for i in range(n):
                x = GRID_X0 + i * GRID_C
                y = GRID_X0 + j * GRID_C
                h = H[j * n + i]
                hx = H[j * n + min(i + 1, n - 1)] - H[j * n + max(i - 1, 0)]
                hy = H[min(j + 1, n - 1) * n + i] - H[max(j - 1, 0) * n + i]
                nx, ny = -hx / (2 * GRID_C), -hy / (2 * GRID_C)
                ln = math.sqrt(nx * nx + ny * ny + 1.0)
                m.vert((x, y, h), (nx / ln, ny / ln, 1.0 / ln), self.terrain_color(x, y, h, 1.0 / ln))
        for j in range(n - 1):
            if j % 20 == 0:
                ld_pump(0.50 + 0.04 * j / n)
            for i in range(n - 1):
                a = j * n + i
                m.T.append((a, a + 1, a + n + 1))
                m.T.append((a, a + n + 1, a + n))
        self.terrain_np = self.render.attachNewNode(m.node('terreno'))

    @staticmethod
    def terrain_color(x, y, h, nz):
        sand = (0.76, 0.70, 0.50)
        if h < 4.0:
            col = sand
        else:
            patch = vnoise(x / 350.0, y / 350.0, 5)
            forest = vnoise(x / 900.0, y / 900.0, 99)
            g = (0.31, 0.53, 0.20) if patch > 0.5 else (0.38, 0.56, 0.22)
            if forest > 0.55:
                g = (0.17, 0.38, 0.15)
            t = smoothstep(2.0, 12.0, h)
            col = tuple(sand[k] * (1 - t) + g[k] * t for k in range(3))
            rock = (0.43, 0.40, 0.37)
            r = max(smoothstep(250, 480, h), smoothstep(0.88, 0.70, nz))
            col = tuple(col[k] * (1 - r) + rock[k] * r for k in range(3))
            s = smoothstep(750, 900, h)
            col = tuple(col[k] * (1 - s) + 0.95 * s for k in range(3))
            dd = 1.0 - smoothstep(5000.0, 11000.0, math.hypot(x - 22500.0, y - 16000.0))   # deserto attorno a Dubai
            if dd > 0.0:
                des = (0.80, 0.69, 0.47) if patch > 0.5 else (0.84, 0.73, 0.50)
                col = tuple(col[k] * (1 - dd) + des[k] * dd for k in range(3))
        return col + (1.0,)

    def build_water(self):
        m = Mesh()
        S, N = 200000.0, 16
        col = (0.10, 0.36, 0.55, 1)
        for j in range(N + 1):
            for i in range(N + 1):
                m.vert((-S / 2 + S * i / N, -S / 2 + S * j / N, 0.0), (0, 0, 1), col)
        for j in range(N):
            for i in range(N):
                a = j * (N + 1) + i
                m.T.append((a, a + 1, a + N + 2))
                m.T.append((a, a + N + 2, a + N + 1))
        self.render.attachNewNode(m.node('mare'))

    # ---- ciclo giorno/notte: 24 ore del gioco = 24 minuti reali (12 di giorno, 12 di notte)
    SKY_KEYS = (   # elevazione del sole (gradi), orizzonte, zenit, luce del sole, luce ambiente
        (90, (0.78, 0.86, 0.94), (0.16, 0.38, 0.78), (1.00, 0.96, 0.88), (0.50, 0.52, 0.58)),
        (22, (0.78, 0.86, 0.94), (0.16, 0.38, 0.78), (1.00, 0.96, 0.88), (0.50, 0.52, 0.58)),
        (9, (0.92, 0.80, 0.64), (0.24, 0.42, 0.76), (1.00, 0.82, 0.60), (0.46, 0.46, 0.54)),
        (2, (1.00, 0.58, 0.30), (0.22, 0.34, 0.62), (0.95, 0.56, 0.30), (0.36, 0.34, 0.44)),
        (-3, (0.66, 0.36, 0.38), (0.12, 0.16, 0.38), (0.30, 0.15, 0.12), (0.22, 0.22, 0.34)),
        (-6, (0.40, 0.26, 0.36), (0.07, 0.10, 0.26), (0.0, 0.0, 0.0), (0.17, 0.18, 0.30)),
        (-14, (0.14, 0.13, 0.24), (0.03, 0.05, 0.15), (0.0, 0.0, 0.0), (0.12, 0.14, 0.25)),
        (-90, (0.05, 0.07, 0.14), (0.01, 0.02, 0.06), (0.0, 0.0, 0.0), (0.10, 0.12, 0.22)))

    def sun_elev(self, tod=None):
        """Elevazione del sole in gradi: sorge alle 6, culmina alle 12, tramonta alle 18."""
        t = self.tod if tod is None else tod
        return 66.0 * math.sin(math.pi * (t - 6.0) / 12.0)

    def env_at(self, e):
        keys = self.SKY_KEYS
        for i in range(len(keys) - 1):
            e0, e1 = keys[i][0], keys[i + 1][0]
            if e0 >= e >= e1:
                f = (e0 - e) / (e0 - e1)
                return [tuple(keys[i][j][k] * (1 - f) + keys[i + 1][j][k] * f for k in range(3)) for j in range(1, 5)]
        return list(keys[-1][1:])

    def build_dome(self, hor, zen, e, az_sun):
        R = 50000.0
        m = Mesh()
        elevs = (-40, -10, 0, 2, 5, 9, 15, 24, 36, 50, 68, 90)
        segs = 36
        tw = clamp(1.0 - abs(e - 1.0) / 15.0, 0.0, 1.0)             # intensita del crepuscolo
        glow = (1.0, 0.50, 0.22) if e > -2.0 else (0.55, 0.28, 0.40)
        rings = []
        for el in elevs:
            er = math.radians(el)
            t = math.sqrt(max(0.0, math.sin(er)))
            base = [hor[k] * (1 - t) + zen[k] * t for k in range(3)]
            ring = []
            for s in range(segs):
                a = 2 * math.pi * s / segs
                amt = tw * max(0.0, math.cos(a - az_sun)) ** 3 * math.exp(-max(el, 0) / 14.0) * 0.85
                col = tuple(base[k] * (1 - amt) + glow[k] * amt for k in range(3)) + (1,)
                ring.append(m.vert((R * math.cos(er) * math.cos(a), R * math.cos(er) * math.sin(a), R * math.sin(er)),
                                   (0, 0, 1), col))
            rings.append(ring)
        for k in range(len(rings) - 1):
            for s in range(segs):
                s2 = (s + 1) % segs
                a, b, c, d = rings[k][s], rings[k][s2], rings[k + 1][s2], rings[k + 1][s]
                m.T.append((a, b, c))
                m.T.append((a, c, d))
        return m

    def build_sky(self):
        R = 50000.0
        self.sky = self.render.attachNewNode('cielo')
        self.sky.setLightOff()
        self.sky.setFogOff()
        self.sky.setTwoSided(True)
        self.sky.setDepthWrite(False)
        self.sky.setDepthTest(False)
        self.dome = None
        ms = Mesh()
        ms.sphere(0, 0, 0, 1500, 1500, 1500, (1, 0.97, 0.85, 1), 10, 6)
        self.sun_disc = self.sky.attachNewNode(ms.node('sole'))
        self.sun_disc.setBin('background', 3)
        mm = Mesh()
        mm.sphere(0, 0, 0, 1100, 1100, 1100, (0.92, 0.94, 1.0, 1), 10, 6)
        self.moon_disc = self.sky.attachNewNode(mm.node('luna'))
        self.moon_disc.setBin('background', 3)
        rng = random.Random(11)
        st = []
        for _ in range(900):
            z = rng.uniform(0.02, 1.0)
            a = rng.uniform(0, 2 * math.pi)
            r = math.sqrt(1 - z * z)
            b = rng.uniform(0.35, 1.0) ** 2
            st.append((R * 0.97 * r * math.cos(a), R * 0.97 * r * math.sin(a), R * 0.97 * z, b, b, min(1.0, b * 1.1), 1.0))
        self.stars = make_points(st, 1.8, self.sky, 'stelle')
        self.stars.setFogOff()
        self.stars.setBin('background', 1)
        self.apply_env(0.0, force=True)

    def apply_env(self, dt, force=False):
        """Aggiorna luci, cielo, nebbia e luminosita notturna in base all'ora."""
        self.env_t += dt
        tod = 11.0 if self.in_hangar else self.tod                  # l'hangar ha sempre luce di giorno
        e = self.sun_elev(tod)
        hor, zen, sun_c, amb_c = self.env_at(e)
        # direzione del sole (verso il sole) e della luna (opposta)
        ph = math.pi * (tod - 6.0) / 12.0
        el = math.radians(e)
        sd = Vec3(math.cos(el) * math.cos(ph), -math.cos(el) * math.sin(ph) * 0.9 + 0.0, math.sin(el))
        sd.normalize()
        md = Vec3(-sd.x, -sd.y * 0.8 + 0.25, -sd.z)
        md.normalize()
        self.night = clamp((4.0 - self.sun_elev()) / 10.0, 0.0, 1.0)
        moon_k = clamp((-e - 1.0) / 12.0, 0.0, 1.0)
        self.amb_np.node().setColor((amb_c[0], amb_c[1], amb_c[2], 1))
        self.sun_np.node().setColor((sun_c[0], sun_c[1], sun_c[2], 1))
        self.moon_np.node().setColor((0.20 * moon_k, 0.25 * moon_k, 0.42 * moon_k, 1))
        self.sun_np.lookAt(Point3(-sd.x, -sd.y, -sd.z))
        self.moon_np.lookAt(Point3(-md.x, -md.y, -md.z))
        self.sun_dir = sd if e >= 0.0 else md          # la luce che da l'ombra all'aereo
        self.shadow_k = clamp(max(clamp(e / 10.0, 0, 1), 0.45 * moon_k), 0.0, 1.0)
        if self.fog is not None:
            self.fog.setColor(*hor)
        self.setBackgroundColor(*hor)
        R = 50000.0
        self.sun_disc.setPos(sd * R * 0.9)
        self.moon_disc.setPos(md * R * 0.9)
        sc = (min(1.0, sun_c[0] * 1.2 + 0.1), min(1.0, sun_c[1] * 1.2 + 0.1), min(1.0, sun_c[2] * 1.3 + 0.05))
        self.sun_disc.setColorScale(sc[0], sc[1], sc[2], 1)
        self.sun_disc.show() if e > -4.0 else self.sun_disc.hide()
        self.moon_disc.show() if md.z > -0.05 else self.moon_disc.hide()
        self.stars.setColorScale(1, 1, 1, clamp((-e - 4.0) / 9.0, 0.0, 1.0))
        self.stars.setH(self.tod * 15.0)
        if force or self.env_t > 0.4:
            self.env_t = 0.0
            if self.dome is not None:
                self.dome.removeNode()
            self.dome = self.sky.attachNewNode(self.build_dome(hor, zen, e, math.atan2(sd.y, sd.x)).node('cupola'))
            self.dome.setBin('background', 0)

    def update_glow(self, t):
        """Luci fisse (piste, finestre, lampioni), luci degli aerei: accensione e lampeggi."""
        nf = 0.0 if self.in_hangar else self.night
        k = nf
        for node, kind in GLOW:
            if node.isEmpty():
                continue
            if kind == 'beacon':
                a = (0.35 + 0.65 * nf) if int(t * 1.3) % 2 == 0 else 0.1
            elif kind == 'blink':
                a = nf * (1.0 if int(t * 0.8) % 2 == 0 else 0.35)
            else:
                a = k
            node.setColorScale(1, 1, 1, a)
        flash = 1.0 if (t % 1.25) < 0.09 or 0.20 < (t % 1.25) < 0.28 else 0.0
        beac = 1.0 if (t % 1.0) < 0.18 else 0.0
        acs = [self.ac] + [rp.ac for rp in self.remotes.values() if rp.ac is not None]
        for ac in acs:
            if ac is None or ac.model is None or ac.model.isEmpty():
                continue
            gl = ac._gl
            if gl is None or gl[0] is not ac.model:
                gl = (ac.model, {n: list(ac.model.findAllMatches('**/' + n))
                                 for n in ('lt_nav', 'lt_strobe', 'lt_beacon', 'lt_land', 'lt_beam', 'lt_cabin')})
                ac._gl = gl
            on = bool(getattr(ac, 'eng', False)) and not ac.crashed
            lg = ac.gear_pos > 0.6 if hasattr(ac, 'gear_pos') else True
            vals = {'lt_nav': nf if not ac.crashed else 0.0,
                    'lt_strobe': flash * (0.3 + 0.7 * nf) if on else 0.0,
                    'lt_beacon': beac * (0.4 + 0.6 * nf) if on else 0.0,
                    'lt_land': nf if (on and lg) else 0.0,
                    'lt_beam': nf * 0.9 if (on and lg) else 0.0,
                    'lt_cabin': nf * (1.0 if on else 0.3) if not ac.crashed else 0.0}
            for n, lst in gl[1].items():
                for nd in lst:
                    if not nd.isEmpty():
                        nd.setColorScale(1, 1, 1, vals[n])

    def build_airports(self):
        ASPH, CONC, PAINT = (0.20, 0.20, 0.22, 1), (0.50, 0.50, 0.50, 1), (0.95, 0.95, 0.95, 1)
        self.gates = {}
        self.gate_msg = ''
        self.pax_tpl = build_pax_templates()
        self.cargo_tpl = build_cargo_templates()
        for _ai, a in enumerate(APTS):
            ld_pump(0.58 + 0.12 * _ai / len(APTS))
            root = self.render.attachNewNode('aeroporto_' + a.code)
            root.setPos(a.cx, a.cy, a.h)
            root.setH(0 if a.hdg == 'N' else -90)
            L, W = a.L, a.W
            mk = Mesh()

            def flat(x0, x1, y0, y1, col, dz=0.12):
                mk.quad([(x0, y0, dz), (x1, y0, dz), (x1, y1, dz), (x0, y1, dz)], col, (x0, y0, -1))

            flat(-W / 2, W / 2, -L / 2, L / 2, ASPH)
            flat(W / 2, 90, -25, 25, ASPH)
            flat(W / 2, 90, -L / 2 + 225, -L / 2 + 275, ASPH)
            flat(90, 290, -700, 700, CONC)
            for y in range(int(-L / 2 + 150), int(L / 2 - 150), 60):
                flat(-0.6, 0.6, y, y + 30, PAINT, 0.14)
            for end, sgn in ((-L / 2 + 20, 1), (L / 2 - 20, -1)):
                for k in range(-6, 7):
                    if k == 0:
                        continue
                    flat(k * 4.0 - 1.0, k * 4.0 + 1.0, end, end + sgn * 40, PAINT, 0.14)
                for dy in (320, 450):
                    flat(-12, -6, end + sgn * dy, end + sgn * (dy + 45), PAINT, 0.14)
                    flat(6, 12, end + sgn * dy, end + sgn * (dy + 45), PAINT, 0.14)
            flat(-W / 2 + 0.5, -W / 2 + 1.0, -L / 2, L / 2, PAINT, 0.14)
            flat(W / 2 - 1.0, W / 2 - 0.5, -L / 2, L / 2, PAINT, 0.14)
            for (p0, p1, q0, q1) in a.pave:                     # raccordi verso le piste parallele
                flat(p0, p1, q0, q1, ASPH)
            for (u0, v0, u1, v1, w) in a.taxi:                  # raccordi obliqui tra le due piste
                dd = math.hypot(u1 - u0, v1 - v0)
                nx, ny = -(v1 - v0) / dd * w / 2, (u1 - u0) / dd * w / 2
                mk.quad([(u0 + nx, v0 + ny, 0.12), (u1 + nx, v1 + ny, 0.12), (u1 - nx, v1 - ny, 0.12), (u0 - nx, v0 - ny, 0.12)],
                        ASPH, (u0, v0, -1))
                for k in range(int(dd // 12)):                  # linea gialla di mezzeria tratteggiata
                    t0, t1 = k * 12.0 / dd, (k * 12.0 + 6.0) / dd
                    cx0, cy0, cx1, cy1 = u0 + (u1 - u0) * t0, v0 + (v1 - v0) * t0, u0 + (u1 - u0) * t1, v0 + (v1 - v0) * t1
                    mk.quad([(cx0 + nx * 0.03, cy0 + ny * 0.03, 0.14), (cx1 + nx * 0.03, cy1 + ny * 0.03, 0.14),
                             (cx1 - nx * 0.03, cy1 - ny * 0.03, 0.14), (cx0 - nx * 0.03, cy0 - ny * 0.03, 0.14)],
                            (0.95, 0.80, 0.10, 1), (cx0, cy0, -1))
            for ri, r in enumerate(a.rws):
                T = _rw_T(r)
                Lr, Wr = r['L'], r['W']

                def fr(x0, x1, y0, y1, col, dz=0.12, T=T):
                    pts = [(T(x0, y0) + (dz,)), (T(x1, y0) + (dz,)), (T(x1, y1) + (dz,)), (T(x0, y1) + (dz,))]
                    mk.quad(pts, col, (pts[0][0], pts[0][1], -1))
                if ri > 0:                                        # piste extra: asfalto, assi e soglie
                    fr(-Wr / 2, Wr / 2, -Lr / 2, Lr / 2, ASPH)
                    for y in range(int(-Lr / 2 + 150), int(Lr / 2 - 150), 60):
                        fr(-0.6, 0.6, y, y + 30, PAINT, 0.14)
                    for end, sgn in ((-Lr / 2 + 20, 1), (Lr / 2 - 20, -1)):
                        for k in range(-5, 6):
                            if k:
                                fr(k * 4.0 - 1.0, k * 4.0 + 1.0, end, end + sgn * 40, PAINT, 0.14)
                    fr(-Wr / 2 + 0.5, -Wr / 2 + 1.0, -Lr / 2, Lr / 2, PAINT, 0.14)
                    fr(Wr / 2 - 1.0, Wr / 2 - 0.5, -Lr / 2, Lr / 2, PAINT, 0.14)
                lb = r['labs']
                if lb and lb[0]:                                  # numeri di pista, leggibili da chi atterra
                    _rw_label(mk, fr, lb[0], -Lr / 2 + 75.0, False)
                    _rw_label(mk, fr, lb[1], Lr / 2 - 75.0, True)
            YEL = (0.95, 0.80, 0.10, 1)
            for gv in GATE_V + CARGO_V:     # stalli di sosta: linea di guida gialla e barra di arresto
                flat(GATE_U - 19.5, GATE_U - 18.9, gv - 30.0, gv + 30.0, YEL, 0.14)
                flat(GATE_U - 24.0, GATE_U - 15.0, gv - 0.3, gv + 0.3, YEL, 0.14)
            flat(95.0, 285.0, 1480.0, 1495.0, (0.95, 0.45, 0.05, 1), 0.14)      # fascia arancio dell'area cargo
            pn = root.attachNewNode(mk.node('pista'))
            pn.setDepthOffset(1)
            mb = Mesh()
            th = a.th
            mb.box(300, 345, -th, th, 0, 16, (0.78, 0.78, 0.82, 1))
            mb.box(298, 347, -th - 2, th + 2, 16, 18, (0.18, 0.26, 0.38, 1))
            for ft in a.feat:
                if ft[0] == 'rot':                              # edificio circolare (come il terminal 1 di Parigi)
                    _, fu, fv, fr_, fh = ft
                    _cyl(mb, fu, fv, 0.0, fh, fr_, (0.80, 0.80, 0.83, 1), 28)
                    _cyl(mb, fu, fv, fh, fh + 2.0, fr_ * 0.6, (0.30, 0.34, 0.40, 1), 28)
                    ox, oy = a.world(fu, fv)
                    add_obstacle(ox, oy, fr_, a.h + fh)
            mb.box(315, 325, 560, 570, 0, 34, (0.75, 0.75, 0.78, 1))
            mb.box(309, 331, 554, 576, 34, 40, (0.15, 0.22, 0.30, 1))
            mb.box(307, 333, 552, 578, 40, 42, (0.6, 0.1, 0.1, 1))
            for k, y in enumerate((-900, -780, 800, 920)):      # hangar con tetto ad arco e portone verso il piazzale
                wall = (0.78, 0.79, 0.83, 1) if k % 2 else (0.82, 0.74, 0.64, 1)
                mb.box(300, 350, y - 50, y + 50, 0, 7, wall)
                mb.box(299.2, 300.0, y - 34, y + 34, 0, 11, (0.22, 0.25, 0.30, 1))
                mb.box(299.0, 299.3, y - 34, y + 34, 0, 1.2, (0.9, 0.75, 0.1, 1))
                ma = Mesh()
                ma.loft([(y - 50, 25, 9, 7), (y + 50, 25, 9, 7)], 16, (0.70, 0.72, 0.77, 1))
                for ye in (y - 50, y + 50):
                    pts = [(25 * math.cos(math.pi * i / 10), ye, 7 + 9 * math.sin(math.pi * i / 10)) for i in range(11)]
                    for i in range(1, 10):
                        ma.tri_flat(pts[0], pts[i], pts[i + 1], wall, (0, y, 7))
                root.attachNewNode(ma.node('arco')).setX(325)
                for dv in (-34, 0, 34):
                    ox, oy = a.world(325, y + dv)
                    add_obstacle(ox, oy, 31, a.h + 16)
            for v in range(-int(th), int(th) + 1, 40):
                ox, oy = a.world(322.5, v)
                add_obstacle(ox, oy, 31, a.h + 18)
            ox, oy = a.world(320, 565)
            add_obstacle(ox, oy, 15, a.h + 42)
            # terminal cargo: capannone con portelloni di carico verso il piazzale
            mb.box(300, 350, CARGO_V[0] - 110, CARGO_V[1] + 110, 0, 14, (0.60, 0.60, 0.62, 1))
            mb.box(298, 352, CARGO_V[0] - 112, CARGO_V[1] + 112, 14, 15.5, (0.92, 0.45, 0.08, 1))
            for cv in CARGO_V:
                mb.box(299.0, 300.2, cv - 12, cv + 12, 0, 8, (0.15, 0.16, 0.20, 1))
                mb.box(298.8, 299.4, cv - 12, cv + 12, 8, 9.5, (0.95, 0.80, 0.10, 1))
                for dv in (-20, 20):
                    mb.box(299.0, 301.0, cv + dv - 4, cv + dv + 4, 0, 6, (0.70, 0.45, 0.20, 1))   # pile di pacchi
            for v in range(int(CARGO_V[0] - 100), int(CARGO_V[1] + 101), 40):
                ox, oy = a.world(325, v)
                add_obstacle(ox, oy, 31, a.h + 15)
            if a.sea:                                           # piattaforma massiccia sotto terminal, piazzali e cargo (la mesh del terreno ha celle da 400 m e qui scende verso il mare)
                vm = max(a.L / 2 + 440.0, 1620.0)
                mb.box(60.0, 580.0, -vm, vm, -a.h - 40.0, -0.10, (0.50, 0.52, 0.48, 1))
            for (pv0, pv1) in getattr(a, 'piers', []):          # tratto di pista sospeso sul mare, su pilastri
                hw = W / 2 + 40.0
                mb.box(-hw, hw, pv0, pv1, -3.0, 0.0, (0.62, 0.62, 0.64, 1))
                for pv in range(int(pv0) + 20, int(pv1), 50):
                    for pu in (-hw + 6, 0.0, hw - 6):
                        mb.box(pu - 2.5, pu + 2.5, pv - 2.5, pv + 2.5, -a.h - 30.0, -3.0, (0.55, 0.55, 0.58, 1))
            root.attachNewNode(mb.node('edifici'))
            self.gates[a.code] = [Bridge(a, root, gv, self.pax_tpl) for gv in GATE_V] + \
                [Bridge(a, root, cv, self.cargo_tpl, 'cargo') for cv in CARGO_V]
            ml = Mesh()      # luci di bordo pista
            for y in range(int(-L / 2), int(L / 2) + 1, 60):
                for sx in (-1, 1):
                    ml.box(sx * (W / 2 + 1.5) - 0.4, sx * (W / 2 + 1.5) + 0.4, y - 0.4, y + 0.4, 0, 1.0,
                           (1.0, 0.95, 0.6, 1))
            green, white, red = (0.2, 1.0, 0.3, 1), (1.0, 1.0, 0.95, 1), (1.0, 0.1, 0.1, 1)
            for sgn in (-1, 1):                  # soglie, luci di avvicinamento e PAPI ai due capi pista
                ve = sgn * L / 2
                for u in range(int(-W / 2), int(W / 2) + 1, 5):
                    ml.box(u - 0.5, u + 0.5, ve - 0.6, ve + 0.6, 0, 0.5, green)
                for d_ in range(30, 630, 30):
                    v = ve + sgn * d_
                    ml.box(-7, 7, v - 0.4, v + 0.4, 0, 0.5, white if d_ < 570 else red)
                    ml.box(-0.5, 0.5, v - 15, v - 14, 0, 0.5, white)
                for i in range(4):
                    ux = -W / 2 - 14 - 8 * i
                    vp = ve - sgn * 300
                    ml.box(ux - 1.5, ux + 1.5, vp - 1.5, vp + 1.5, 0, 1.4, white if i < 2 else red)
            for r in a.rws[1:]:                  # luci di bordo e soglie delle piste extra
                T = _rw_T(r)
                for y in range(int(-r['L'] / 2), int(r['L'] / 2) + 1, 60):
                    for sx in (-1, 1):
                        px, py = T(sx * (r['W'] / 2 + 1.5), y)
                        ml.box(px - 0.4, px + 0.4, py - 0.4, py + 0.4, 0, 1.0, (1.0, 0.95, 0.6, 1))
                for sgn in (-1, 1):
                    for x in range(int(-r['W'] / 2), int(r['W'] / 2) + 1, 6):
                        px, py = T(x, sgn * r['L'] / 2)
                        ml.box(px - 0.5, px + 0.5, py - 0.5, py + 0.5, 0, 0.5, green if sgn < 0 else red)
            ln = root.attachNewNode(ml.node('luci'))
            ln.setLightOff()
            self.night_lights(a, root, th)

    def night_lights(self, a, root, th):
        """Luci notturne dell'aeroporto: piste, avvicinamento, PAPI, raccordi blu, piazzale, terminal, torre."""
        run, tax, win, flood, stand = [], [], [], [], []
        W_ = (1.0, 0.93, 0.66)
        for r in a.rws:
            T = _rw_T(r)
            Lr, Wr = r['L'], r['W']
            for y in range(int(-Lr / 2), int(Lr / 2) + 1, 30):
                for sx in (-1, 1):
                    run.append(T(sx * (Wr / 2 + 1.5), y) + (0.9,) + W_ + (1.0,))
            for y in range(int(-Lr / 2) + 15, int(Lr / 2) - 14, 20):
                run.append(T(0.0, y) + (0.3, 0.95, 0.97, 1.0, 0.8))
            for sgn in (-1, 1):
                ve = sgn * Lr / 2
                for x in range(int(-Wr / 2), int(Wr / 2) + 1, 3):
                    run.append(T(x, ve) + (0.5, 0.15, 1.0, 0.25, 1.0))          # soglia verde
                    run.append(T(x, ve - sgn * 1.5) + (0.5, 1.0, 0.1, 0.1, 0.8))   # fine pista rossa
                for d_ in range(30, 630, 30):
                    run.append(T(0.0, ve + sgn * d_) + (0.8, 1.0, 1.0, 0.95, 1.0))
                    if d_ in (150, 300, 450):
                        for x in range(-15, 16, 3):
                            run.append(T(float(x), ve + sgn * d_) + (0.8, 1.0, 1.0, 0.95, 0.9))
                for i in range(4):
                    run.append(T(-Wr / 2 - 14 - 8 * i, ve - sgn * 300) + (1.2,) + ((1, 1, 0.95) if i < 2 else (1, 0.1, 0.1)) + (1.0,))
        blue = (0.2, 0.35, 1.0)
        strips = [(a.W / 2, 90.0, -25.0, 25.0), (a.W / 2, 90.0, -a.L / 2 + 225.0, -a.L / 2 + 275.0)] + list(
            (p0, p1, q0, q1) for (p0, p1, q0, q1) in a.pave)
        for (p0, p1, q0, q1) in strips:
            for u in range(int(p0), int(p1) + 1, 30):
                tax.append((u, q0, 0.5) + blue + (1.0,))
                tax.append((u, q1, 0.5) + blue + (1.0,))
        for v in range(-700, 701, 35):                                  # contorno del piazzale
            tax.append((90.0, v, 0.5) + blue + (1.0,))
            tax.append((290.0, v, 0.5) + blue + (1.0,))
        for u in range(90, 291, 35):
            tax.append((u, -700.0, 0.5) + blue + (1.0,))
            tax.append((u, 700.0, 0.5) + blue + (1.0,))
        for gv in GATE_V:                                                # linee di guida dei gate
            for dv in range(-30, 31, 5):
                stand.append((GATE_U - 19.2, gv + dv, 0.3, 1.0, 0.85, 0.1, 1.0))
        for v in range(-int(th), int(th) + 1, 7):                       # finestre del terminal
            for z in (4.0, 10.0):
                if (v // 7 + int(z)) % 5:
                    win.append((298.6, v, z, 1.0, 0.85, 0.52, 0.95))
        mf = Mesh()
        for v in range(-600, 601, 200):                                 # torri faro del piazzale
            mf.box(199.6, 200.4, v - 0.4, v + 0.4, 0.0, 22.0, (0.4, 0.4, 0.42, 1))
            flood.append((200.0, v, 22.5, 1.0, 0.95, 0.78, 1.0))
            flood.append((200.0, v, 21.5, 1.0, 0.95, 0.78, 1.0))
        root.attachNewNode(mf.node('torri_faro'))
        for node, size, kind in ((make_points(run, 4.5, root, 'luci_pista'), 4.5, 'steady'),
                                 (make_points(tax, 3.5, root, 'luci_raccordi'), 3.5, 'steady'),
                                 (make_points(stand, 3.0, root, 'luci_stalli'), 3.0, 'steady'),
                                 (make_points(win, 3.5, root, 'finestre'), 3.5, 'steady'),
                                 (make_points(flood, 9.0, root, 'fari_piazzale'), 9.0, 'steady'),
                                 (make_points([(320.0, 565.0, 43.0, 1.0, 0.1, 0.1, 1.0)], 12.0, root, 'torre'), 12.0, 'beacon'),
                                 (make_points([(a2, 565.0 + b2, 36.5, 0.7, 1.0, 1.0, 0.9) for a2 in (309.5, 330.5)
                                               for b2 in (-10.5, -3.5, 3.5, 10.5)], 5.0, root, 'cabina_torre'), 5.0, 'steady')):
            GLOW.append((node, kind))

    def build_trees_and_houses(self):
        rng = random.Random(7)
        mt = Mesh()
        count = tries = 0
        while count < 9000 and tries < 90000:
            tries += 1
            if tries % 400 == 0:
                ld_pump()
            if tries % 2:
                x, y = rng.uniform(-MAP_R + 6000, MAP_R - 6000), rng.uniform(-MAP_R + 6000, MAP_R - 6000)
            else:
                a = rng.choice(APTS)
                ang, rr = rng.uniform(0, 2 * math.pi), rng.uniform(600, 7000)
                x, y = a.cx + rr * math.cos(ang), a.cy + rr * math.sin(ang)
            gh, raw = self.terrain.sample(x, y)
            if raw < 8 or raw > 650 or near_airport(x, y, 350) or near_city(x, y):
                continue
            if math.hypot(x - 22500.0, y - 16000.0) < 8000.0 and rng.random() < 0.97:     # deserto di Dubai
                continue
            if vnoise(x / 700.0, y / 700.0, 99) < 0.5:
                continue
            if abs(self.terrain.sample(x + 30, y)[0] - gh) > 14 or abs(self.terrain.sample(x, y + 30)[0] - gh) > 14:
                continue
            hgt = rng.uniform(11, 20)
            gv = rng.uniform(0.75, 1.15)
            mt.cone(x, y, gh - 0.5, hgt * 0.32, hgt, 6, (0.10 * gv, 0.32 * gv, 0.12 * gv, 1))
            count += 1
        self.render.attachNewNode(mt.node('alberi'))

        mh = Mesh()
        hl = []
        walls = ((0.92, 0.88, 0.78, 1), (0.88, 0.70, 0.55, 1), (0.85, 0.85, 0.88, 1), (0.9, 0.78, 0.6, 1))
        centers = []
        tries = 0
        for a in APTS:       # due cittadine vicino a ogni aeroporto
            for _ in range(2):
                ang, rr = rng.uniform(0, 2 * math.pi), rng.uniform(2200, 5000)
                centers.append((a.cx + rr * math.cos(ang), a.cy + rr * math.sin(ang)))
        while len(centers) < 20 and tries < 40000:
            tries += 1
            ang, rr = rng.uniform(0, 2 * math.pi), rng.uniform(3000, MAP_R - 14000)
            centers.append((rr * math.cos(ang), rr * math.sin(ang)))
        cb = CityBuilder(self.terrain)
        for cx, cy in centers:
            ld_pump()
            gh, raw = self.terrain.sample(cx, cy)
            if raw < 10 or raw > 250 or near_airport(cx, cy, 400) or near_city(cx, cy, 300) or in_corridor(cx, cy, 2300.0):
                continue
            if any(abs(self.terrain.sample(cx + dx, cy + dy)[0] - gh) > 6 for dx in (-150, 150) for dy in (-150, 150)):
                continue
            ld_pump()
            rd = cb.downtown(cx, cy, rng)           # grattacieli nel centro
            for _ in range(26):
                hx, hy = cx + rng.uniform(-300, 300), cy + rng.uniform(-300, 300)
                if math.hypot(hx - cx, hy - cy) < rd + 14:
                    continue
                hg = self.terrain.sample(hx, hy)[0]
                w, d, h = rng.uniform(8, 14), rng.uniform(8, 12), rng.uniform(5, 9)
                mh.box(hx - w / 2, hx + w / 2, hy - d / 2, hy + d / 2, hg - 3, hg + h, rng.choice(walls))
                mh.box(hx - w / 2 - 0.5, hx + w / 2 + 0.5, hy - d / 2 - 0.5, hy + d / 2 + 0.5, hg + h, hg + h + 1.2,
                       (0.62, 0.22, 0.15, 1))
                if rng.random() < 0.7:                  # una finestra accesa per casa
                    hl.append((hx + w / 2 + 0.4, hy + rng.uniform(-d / 3, d / 3), hg + h * 0.5, 1.0, 0.8, 0.45, 0.9))
                if rng.random() < 0.5:
                    hl.append((hx - w / 2 - 0.4, hy + rng.uniform(-d / 3, d / 3), hg + h * 0.5, 1.0, 0.85, 0.55, 0.9))
        self.render.attachNewNode(mh.node('case'))
        if hl:
            GLOW.append((make_points(hl, 2.4, self.render, 'luci_case'), 'steady'))
        ld_pump()
        cb.metropolis(rng)                          # la citta moderna con il Burj Khalifa
        cb.finish(self.render)
        self.beacon_np = getattr(cb, 'beacons', None)

    def build_clouds(self):
        rng = random.Random(3)
        m = Mesh()
        for _ in range(260):
            cx, cy = rng.uniform(-MAP_R + 5000, MAP_R - 5000), rng.uniform(-MAP_R + 5000, MAP_R - 5000)
            cz = rng.uniform(900, 2800)
            s = rng.uniform(150, 380)
            for _ in range(rng.randint(5, 8)):
                m.sphere(cx + rng.uniform(-2, 2) * s, cy + rng.uniform(-1.2, 1.2) * s, cz + rng.uniform(-0.2, 0.3) * s,
                         s * rng.uniform(0.8, 1.4), s * rng.uniform(0.7, 1.1), s * rng.uniform(0.4, 0.6),
                         (0.97, 0.97, 0.99, 1), 8, 5)
        self.render.attachNewNode(m.node('nuvole'))

    # ---- multiplayer: pannello, stato locale, aerei remoti
    def build_mp_ui(self):
        f = DirectFrame(parent=self.aspect2d, frameColor=(0.03, 0.05, 0.09, 0.94), frameSize=(-0.95, 0.95, -0.86, 0.72))
        self.mp_ui = f
        kw = dict(parent=f, align=TextNode.ALeft, fg=(1, 0.85, 0.3, 1))
        OnscreenText(text='MULTIPLAYER', pos=(-0.88, 0.60), scale=0.085, **dict(kw, fg=(1, 1, 1, 1)))
        OnscreenText(text='Il tuo nome', pos=(-0.88, 0.48), scale=0.045, **kw)
        ek = dict(parent=f, scale=0.05, numLines=1, focus=0, frameColor=(0.14, 0.17, 0.25, 1), text_fg=(1, 1, 1, 1),
                  relief=DGG.FLAT, suppressKeys=1)
        self.mp_e_name = DirectEntry(pos=(-0.88, 0, 0.41), width=10, initialText=self.net.name, **ek)
        OnscreenText(text='1) CREA una partita e dai il codice agli amici', pos=(-0.88, 0.28), scale=0.045, **kw)
        self._mp_btn('CREA PARTITA', (-0.55, 0, 0.19), self.mp_host, (0.15, 0.5, 0.2, 1))
        self._mp_btn('COPIA CODICE', (0.10, 0, 0.19), self.mp_copy, (0.2, 0.3, 0.5, 1))
        self.mp_code = OnscreenText(text='', pos=(-0.88, 0.07), scale=0.055, mayChange=True, **dict(kw, fg=(0.6, 1, 0.8, 1)))
        OnscreenText(text='2) oppure ENTRA con il codice di un amico', pos=(-0.88, -0.07), scale=0.045, **kw)
        self.mp_e_code = DirectEntry(pos=(-0.88, 0, -0.15), width=22, **ek)
        self._mp_btn('ENTRA', (0.70, 0, -0.13), self.mp_join, (0.15, 0.5, 0.2, 1), 0.2)
        self.mp_status = OnscreenText(text='', pos=(-0.88, -0.28), scale=0.045, mayChange=True,
                                      **dict(kw, fg=(0.85, 0.95, 1, 1)))
        self.mp_list = OnscreenText(text='', pos=(-0.88, -0.37), scale=0.042, mayChange=True,
                                    **dict(kw, fg=(0.9, 0.9, 0.9, 1)))
        self.mp_upnp = OnscreenText(text='', pos=(-0.88, -0.46), scale=0.034, mayChange=True, wordwrap=52,
                                    **dict(kw, fg=(0.95, 0.85, 0.5, 1)))
        OnscreenText(text='Ogni giocatore usa il proprio PC con FlightWorld installato. Stessa rete: basta il codice. '
                     'Reti diverse: l\'host apre la porta UDP %d in automatico (UPnP) oppure usate una VPN. '
                     'Al primo avvio consenti FlightWorld nel Firewall di Windows. Se due aerei si scontrano in volo esplodono.'
                     % NET_PORT, pos=(-0.88, -0.62), scale=0.030, wordwrap=60, **dict(kw, fg=(0.65, 0.7, 0.8, 1)))
        self._mp_btn('ESCI DALLA PARTITA', (-0.45, 0, -0.79), self.mp_leave, (0.5, 0.15, 0.15, 1), 0.38)
        self._mp_btn('CHIUDI', (0.6, 0, -0.79), self.hide_mp, (0.2, 0.3, 0.5, 1), 0.3)
        DirectFrame(parent=f, frameColor=(1, 0.78, 0.25, 1), frameSize=(-0.95, 0.95, 0.713, 0.72))
        DirectFrame(parent=f, frameColor=(1, 0.78, 0.25, 1), frameSize=(-0.95, 0.95, -0.86, -0.853))
        self.ui_hover(f)
        f.hide()

    def _mp_btn(self, text, pos, cmd, col, half=0.3):
        return DirectButton(parent=self.mp_ui, text=text, text_scale=0.045, text_fg=(1, 1, 1, 1), pos=pos,
                            frameSize=(-half, half, -0.045, 0.065), frameColor=col, relief=DGG.FLAT,
                            pressEffect=0, command=cmd)

    def show_mp(self):
        self.mp_ui.show()
        self.ui_in(self.mp_ui, 0.0, -0.2, 0.45, 0.0, True, 0.94)
        self.refresh_mp()

    def hide_mp(self):
        self.mp_ui.hide()

    def _mp_name(self):
        nm = re.sub(r'[^\w .-]', '', self.mp_e_name.get())[:14].strip().upper()
        self.net.name = nm or 'PILOTA'
        return self.net.name

    def mp_host(self):
        self.net.start_host(self._mp_name())
        self.refresh_mp()

    def mp_join(self):
        self.net.start_client(self.mp_e_code.get(), self._mp_name())
        self.refresh_mp()

    def mp_copy(self):
        if self.net.role != 'host':
            return
        try:
            subprocess.run(['clip'], input=self.net.code().encode('utf-16'), timeout=3,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.net.status = 'Codice copiato negli appunti'
        except Exception:
            self.net.status = 'Copia non riuscita: scrivi il codice a mano'

    def mp_leave(self):
        self.net.stop()
        self.net.status = 'Sei uscito dalla partita'
        self.refresh_mp()

    def refresh_mp(self):
        n = self.net
        self.mp_code.setText(('CODICE: ' + n.code() + ('' if n.pub_ip else '   (rete locale: %s)' % n.local_code()))
                             if n.role == 'host' else '')
        self.mp_status.setText(n.status)
        self.mp_upnp.setText(n.upnp if n.role == 'host' else '')
        self.mp_list.setText('Giocatori (%d): %s' % (len(n.players()), ', '.join(n.players())) if n.role else '')

    def my_state(self):
        ac = self.ac
        sp = ac.sp
        c = CUSTOM.get(sp.short, {})
        dn, dr = default_text(sp)
        r = lambda v: [round(x, 2) for x in v]
        return dict(s=SPECS.index(sp), c=[list(c.get('body', sp.base['body'])), list(c.get('accent', sp.base['accent'])),
                                          c.get('name') or dn, c.get('reg') or dr, int(c.get('livery', 0))],
                    p=r(ac.pos), f=[round(x, 4) for x in ac.fwd], u=[round(x, 4) for x in ac.up], v=r(ac.vel),
                    g=round(ac.gear_pos, 2), th=round(ac.eff_throttle(), 2), fl=ac.flaps, cr=ac.crashed, og=ac.on_ground,
                    hd=bool(self.in_menu), ct=[round(self.ctl.elev, 2), round(self.ctl.ail, 2), round(self.ctl.rud, 2)])

    def mp_update(self, dt):
        net = self.net
        if net.role is None:
            if self.remotes:
                for rp in self.remotes.values():
                    rp.destroy()
                self.remotes = {}
            self.t_mp.setText('')
            return
        now = time.time()
        net.poll(now)
        if (net.joined or net.role == 'client') and not self.in_hangar:
            net.send_state(self.my_state(), now)
        live = {i: (s, now - t) for i, (s, t) in net.states.items() if now - t < 3.0 and i != net.my_id}
        for i in [i for i in self.remotes if i not in live]:
            self.remotes.pop(i).destroy()
        for i, (st, age) in live.items():
            if i not in self.remotes:
                self.remotes[i] = RemotePlane(self, st)
            self.remotes[i].update(st, age, dt)
        me = self.ac
        if not self.in_menu and not self.paused and not me.crashed and not me.ditched and not me.on_ground:
            for rp in self.remotes.values():                 # scontro in volo: esplodono
                if rp.ac and not rp.crashed and planes_collide(me, rp.ac):
                    me.crash('collisione in volo con %s' % (rp.label.node().getText().split('  ')[0] if rp.label else 'un altro aereo'))
                    break
        names = net.players()
        self.t_mp.setText('' if self.in_hangar else 'MULTIPLAYER %s: %d giocatori%s' % (
            'host' if net.role == 'host' else 'client', len(names), '' if net.joined else '  (non collegato)'))
        if self.mp_ui.isHidden() is False:
            self.refresh_mp()

    # ---- hangar: sala di personalizzazione lontana dal mondo, sopra il mare
    HANGAR_POS = Vec3(75000.0, 0.0, 1500.0)
    PALETTE = ((0.95, 0.95, 0.97), (0.10, 0.28, 0.62), (0.78, 0.10, 0.10), (0.95, 0.50, 0.10), (0.10, 0.50, 0.25),
               (0.35, 0.22, 0.55), (0.08, 0.08, 0.10), (0.95, 0.80, 0.15), (0.45, 0.50, 0.55), (0.0, 0.60, 0.70))
    ENGINES = ('STANDARD', 'POTENZIATO +25%', 'RACING +60%')
    WEIGHTS = ('NORMALE', 'LEGGERO -10%', 'PESANTE +15%')
    SHORT = ('C172', 'B737', 'A320', 'A380', 'CONC', 'JET', 'FEDEX', 'DHL', 'AMZ')

    def build_hangar(self):
        hp = self.HANGAR_POS
        self.hangar_np = self.render.attachNewNode('hangar')
        self.hangar_np.setPos(hp)
        HX, HY, HZ = 140, 140, 60
        m, lm = Mesh(), Mesh()
        room = Vec3(0, 0, HZ / 2)

        def inq(pts, col):
            qc = sum((Vec3(*p) for p in pts), Vec3(0, 0, 0)) / 4.0
            m.quad(pts, col, tuple(qc * 2 - room))

        n = int(2 * HX / 10)
        for i in range(n):
            for j in range(n):
                x0, y0 = -HX + i * 10, -HY + j * 10
                col = (0.52, 0.53, 0.55, 1) if (i + j) % 2 else (0.47, 0.48, 0.50, 1)
                inq([(x0, y0, 0), (x0 + 10, y0, 0), (x0 + 10, y0 + 10, 0), (x0, y0 + 10, 0)], col)
        for i in range(n):
            a, b = -HX + i * 10, -HX + (i + 1) * 10
            tone = 0.80 + 0.04 * (i % 2)
            for (p0, p1) in (((a, -HY), (b, -HY)), ((a, HY), (b, HY)), ((-HX, a), (-HX, b)), ((HX, a), (HX, b))):
                inq([(p0[0], p0[1], 0), (p1[0], p1[1], 0), (p1[0], p1[1], 7), (p0[0], p0[1], 7)], (0.20, 0.27, 0.40, 1))
                inq([(p0[0], p0[1], 7), (p1[0], p1[1], 7), (p1[0], p1[1], HZ), (p0[0], p0[1], HZ)],
                    (tone, tone + 0.02, tone + 0.06, 1))
            inq([(a, -HY, HZ), (b, -HY, HZ), (b, HY, HZ), (a, HY, HZ)], (0.86, 0.87, 0.90, 1))
        for t in range(-HX + 20, HX, 20):
            m.box(t - 0.5, t + 0.5, -HY, HY, HZ - 3.0, HZ - 2.0, (0.30, 0.32, 0.36, 1))
            m.box(-HX, HX, t - 0.5, t + 0.5, HZ - 2.0, HZ - 1.0, (0.26, 0.28, 0.32, 1))
            m.box(-HX, -HX + 1.4, t - 0.6, t + 0.6, 0, HZ, (0.62, 0.64, 0.68, 1))
            m.box(HX - 1.4, HX, t - 0.6, t + 0.6, 0, HZ, (0.62, 0.64, 0.68, 1))
            m.box(t - 0.6, t + 0.6, -HY, -HY + 1.4, 0, HZ, (0.62, 0.64, 0.68, 1))
            m.box(t - 0.6, t + 0.6, HY - 1.4, HY, 0, HZ, (0.62, 0.64, 0.68, 1))
        for x in (-90, -45, 0, 45, 90):                   # plafoniere
            lm.box(x - 1.6, x + 1.6, -120, 120, HZ - 4.4, HZ - 4.1, (1.0, 1.0, 0.94, 1))
        # portone sul fondo con righe di sicurezza e linee gialle sul pavimento
        m.box(-60, 60, HY - 1.8, HY - 1.4, 0, 40, (0.20, 0.22, 0.26, 1))
        for i in range(12):
            m.box(-60 + i * 10, -55 + i * 10, HY - 2.2, HY - 1.8, 0, 3, (0.95, 0.78, 0.05, 1) if i % 2 else (0.1, 0.1, 0.1, 1))
        for (x0, x1, y0, y1) in ((-120, 120, -120, -119), (-120, 120, 119, 120), (-120, -119, -120, 120), (119, 120, -120, 120)):
            m.box(x0, x1, y0, y1, 0.0, 0.08, (0.95, 0.78, 0.05, 1))
        # attrezzi: carrelli rossi, bidoni, pile di gomme, trattore di traino
        for (x, y) in ((-125, -60), (-125, 20), (-125, 80), (120, -90), (120, 40)):
            m.box(x - 2, x + 2, y - 1.2, y + 1.2, 0, 2.2, (0.72, 0.10, 0.10, 1))
            m.box(x - 2.1, x + 2.1, y - 1.3, y + 1.3, 2.2, 2.5, (0.18, 0.18, 0.2, 1))
        for (x, y) in ((-110, 125), (-104, 125), (-98, 122), (100, 125), (106, 122)):
            m.sphere(x, y, 1.0, 0.9, 0.9, 1.0, (0.2, 0.35, 0.7, 1), 8, 5)
        for (x, y) in ((110, -122), (113, -122), (110, -118)):
            for k in range(4):
                m.box(x - 1.0, x + 1.0, y - 1.0, y + 1.0, k * 0.5, k * 0.5 + 0.5, (0.08, 0.08, 0.09, 1))
        m.box(-30, -24, -125, -120, 0.5, 2.5, (0.95, 0.75, 0.1, 1))
        m.box(-31, -23, -126, -119, 2.5, 3.2, (0.2, 0.2, 0.22, 1))
        self.hangar_np.attachNewNode(m.node('hangar_sala'))
        self.hangar_np.attachNewNode(lm.node('hangar_luci')).setLightOff()
        for x in (-45.0, 45.0):                           # luci puntuali solo per la sala e l'aereo
            pl = PointLight('hl%d' % int(x))
            pl.setColor((2.3, 2.2, 2.05, 1))
            pl.setAttenuation((1.0, 0.0, 0.0005))
            pn = self.hangar_np.attachNewNode(pl)
            pn.setPos(x, 0, HZ - 8)
            self.hangar_np.setLight(pn)
            self.hangar_lights = getattr(self, 'hangar_lights', []) + [pn]
        self.hangar_np.setLightOff(self.sun_np)
        self.hangar_np.hide()

    def build_hangar_ui(self):
        f = DirectFrame(parent=self.a2dLeftCenter, frameColor=(0.03, 0.05, 0.09, 0.88), frameSize=(0, 1.15, -0.97, 0.97),
                        pos=(0.02, 0, 0))
        self.hui = f
        kw = dict(parent=f, align=TextNode.ALeft, fg=(1, 0.85, 0.3, 1))
        OnscreenText(text='HANGAR', pos=(0.04, 0.86), scale=0.085, **dict(kw, fg=(1, 1, 1, 1)))
        self.h_title = OnscreenText(text='', pos=(0.04, 0.79), scale=0.048, mayChange=True, **dict(kw, fg=(0.8, 0.9, 1, 1)))
        OnscreenText(text='AEREO', pos=(0.04, 0.70), scale=0.045, **kw)
        self.h_btn_ac = []
        for i, nm in enumerate(self.SHORT):
            self.h_btn_ac.append(DirectButton(parent=f, text=nm, text_scale=0.030, text_fg=(1, 1, 1, 1), pos=(0.075 + 0.124 * i, 0, 0.64),
                                              frameSize=(-0.058, 0.058, -0.04, 0.06), relief=DGG.FLAT, pressEffect=0,
                                              command=self.h_pick_ac, extraArgs=[i]))
        self.h_sw = {}
        for key, label, y in (('body', 'COLORE CORPO', 0.52), ('accent', 'COLORE FASCIA, CODA, SCRITTA', 0.30)):
            OnscreenText(text=label, pos=(0.04, y + 0.07), scale=0.045, **kw)
            row = []
            for i, col in enumerate(self.PALETTE):
                row.append(DirectButton(parent=f, text='', text_scale=0.07, text_fg=(0.5, 0.5, 0.5, 1), text_pos=(0, -0.025),
                                        pos=(0.09 + 0.105 * i, 0, y - 0.01), frameSize=(-0.045, 0.045, -0.045, 0.045),
                                        frameColor=col + (1,), relief=DGG.FLAT, pressEffect=0,
                                        command=self.h_set_color, extraArgs=[key, col]))
            self.h_sw[key] = row
        OnscreenText(text='SCRITTA E MATRICOLA  (Invio per applicare)', pos=(0.04, 0.17), scale=0.042, **kw)
        ek = dict(parent=f, scale=0.05, width=20, numLines=1, focus=0, frameColor=(0.14, 0.17, 0.25, 1), text_fg=(1, 1, 1, 1),
                  relief=DGG.FLAT, suppressKeys=1)
        self.h_e_name = DirectEntry(pos=(0.04, 0, 0.09), command=self.h_text, **ek)
        self.h_e_reg = DirectEntry(pos=(0.04, 0, 0.01), command=self.h_text, **ek)
        OnscreenText(text='MOTORE', pos=(0.04, -0.10), scale=0.045, **kw)
        self.h_btn_pw = [DirectButton(parent=f, text=t, text_scale=0.034, text_fg=(1, 1, 1, 1), pos=(0.18 + 0.35 * i, 0, -0.17),
                                      frameSize=(-0.17, 0.17, -0.04, 0.06), relief=DGG.FLAT, pressEffect=0,
                                      command=self.h_set_level, extraArgs=['power', i]) for i, t in enumerate(self.ENGINES)]
        OnscreenText(text='PESO', pos=(0.04, -0.27), scale=0.045, **kw)
        self.h_btn_wt = [DirectButton(parent=f, text=t, text_scale=0.034, text_fg=(1, 1, 1, 1), pos=(0.18 + 0.35 * i, 0, -0.34),
                                      frameSize=(-0.17, 0.17, -0.04, 0.06), relief=DGG.FLAT, pressEffect=0,
                                      command=self.h_set_level, extraArgs=['weight', i]) for i, t in enumerate(self.WEIGHTS)]
        self.h_stats = OnscreenText(text='', pos=(0.04, -0.47), scale=0.040, mayChange=True, **dict(kw, fg=(0.85, 0.95, 1, 1)))
        self.h_lv_label = OnscreenText(text='LIVREA COMPAGNIA', pos=(0.04, -0.585), scale=0.045, **kw)
        self.h_lv_btns = [DirectButton(parent=f, text=t, text_scale=0.06, text_fg=(1, 1, 1, 1), pos=(x, 0, -0.665),
                                       frameSize=(-0.07, 0.07, -0.05, 0.075), frameColor=(0.2, 0.3, 0.5, 1), relief=DGG.FLAT,
                                       pressEffect=0, command=self.h_livery, extraArgs=[d])
                          for t, x, d in (('<', 0.09, -1), ('>', 1.06, 1))]
        self.h_lv_name = OnscreenText(text='', pos=(0.575, -0.68), scale=0.047, mayChange=True, parent=f,
                                      align=TextNode.ACenter, fg=(1, 1, 1, 1))
        OnscreenText(text='Frecce sinistra/destra: gira   su/giu: zoom', pos=(0.04, -0.765), scale=0.034,
                     **dict(kw, fg=(0.7, 0.75, 0.85, 1)))
        for txt, x, cmd, col in (('RIPRISTINA', 0.28, self.h_reset, (0.45, 0.2, 0.15, 1)),
                                 ('FATTO: TORNA AL MENU', 0.80, self.leave_hangar, (0.15, 0.5, 0.2, 1))):
            DirectButton(parent=f, text=txt, text_scale=0.045, text_fg=(1, 1, 1, 1), pos=(x, 0, -0.86),
                         frameSize=(-0.26 if x < 0.5 else -0.34, 0.26 if x < 0.5 else 0.34, -0.05, 0.07),
                         frameColor=col, relief=DGG.FLAT, pressEffect=0, command=cmd)
        f.hide()

    def refresh_hangar(self):
        sp = SPECS[self.sel_spec]
        c = CUSTOM.get(sp.short, {})
        self.h_title.setText(sp.name)
        for i, b in enumerate(self.h_btn_ac):
            b['frameColor'] = (0.2, 0.5, 0.9, 1) if i == self.sel_spec else (0.15, 0.17, 0.25, 0.95)
        for key in ('body', 'accent'):
            cur = tuple(round(v, 2) for v in c.get(key, sp.base[key]))
            for col, b in zip(self.PALETTE, self.h_sw[key]):
                b['text'] = 'O' if tuple(round(v, 2) for v in col) == cur else ''
                b['text_fg'] = (0.5, 0.5, 0.5, 1) if sum(col) > 1.5 else (1, 1, 1, 1)
        for i, b in enumerate(self.h_btn_pw):
            b['frameColor'] = (0.2, 0.5, 0.9, 1) if i == c.get('power', 0) else (0.15, 0.17, 0.25, 0.95)
        for i, b in enumerate(self.h_btn_wt):
            b['frameColor'] = (0.2, 0.5, 0.9, 1) if i == c.get('weight', 0) else (0.15, 0.17, 0.25, 0.95)
        dn, dr = default_text(sp)
        self.h_e_name.enterText(c.get('name') or dn)
        self.h_e_reg.enterText(c.get('reg') or dr)
        has_lv = sp.geo is not None and not sp.cargo
        for w in [self.h_lv_label, self.h_lv_name] + self.h_lv_btns:
            w.show() if has_lv else w.hide()
        lv = LIVERIES[c.get('livery', 0) if 0 <= c.get('livery', 0) < len(LIVERIES) else 0]
        self.h_lv_name.setText(lv['label'])
        thrust = sp.T0 if sp.kind == 'jet' else sp.P / 40.0
        self.h_stats.setText('Massa %d kg   Spinta %d kN\nSpinta/peso %.2f   Rotazione %d km/h' % (
            sp.mass, thrust / 1000.0, thrust / (sp.mass * G), sp.vr * 3.6 * 1.1))

    def rebuild_in_hangar(self):
        for sp in SPECS:
            _TEX_CACHE.pop(sp.short, None)
        _TEX_CACHE.pop('cessna', None)
        _TEX_CACHE.pop('concorde', None)
        self.new_aircraft(self.sel_spec)
        self.hangar_light_aircraft()
        self.refresh_hangar()

    def hangar_light_aircraft(self):
        if self.in_hangar:
            self.ac.model.setLightOff(self.sun_np)
            for pn in self.hangar_lights:
                self.ac.model.setLight(pn)

    def h_pick_ac(self, i):
        self.sel_spec = i
        self.rebuild_in_hangar()

    def h_livery(self, d):
        sp = SPECS[self.sel_spec]
        if sp.geo is None or sp.cargo:
            return
        c = CUSTOM.setdefault(sp.short, {})
        new = (c.get('livery', 0) + d) % len(LIVERIES)
        c['livery'] = new
        if new:
            c['name'] = LIVERIES[new]['name']
        else:
            c.pop('name', None)
        apply_custom(sp)
        save_custom()
        self.rebuild_in_hangar()

    def h_set_color(self, key, col):
        sp = SPECS[self.sel_spec]
        if sp.cargo:
            return
        c_ = CUSTOM.setdefault(sp.short, {})
        if c_.get('livery'):                       # scegliendo un colore a mano si torna alla livrea personalizzata
            c_['body'] = tuple(livery_of(c_)['body'])
            c_['accent'] = tuple(livery_of(c_)['tail'])
            c_['livery'] = 0
        c_[key] = tuple(col)
        apply_custom(sp)
        save_custom()
        self.rebuild_in_hangar()

    def h_set_level(self, key, i):
        sp = SPECS[self.sel_spec]
        CUSTOM.setdefault(sp.short, {})[key] = i
        apply_custom(sp)
        save_custom()
        self.rebuild_in_hangar()

    def h_text(self, _=None):
        sp = SPECS[self.sel_spec]
        c = CUSTOM.setdefault(sp.short, {})
        c['name'] = self.h_e_name.get().upper()[:18]
        c['reg'] = self.h_e_reg.get().upper()[:12]
        apply_custom(sp)
        save_custom()
        self.rebuild_in_hangar()

    def h_reset(self):
        sp = SPECS[self.sel_spec]
        CUSTOM.pop(sp.short, None)
        apply_custom(sp)
        save_custom()
        self.rebuild_in_hangar()

    def enter_hangar(self):
        self.in_hangar = True
        self.in_menu = True
        self.menu.hide()
        TEX_K[0] = 1                       # anteprima veloce
        self.fx.clear(self.ac)
        self.hangar_np.show()
        self.hui.show()
        self.ui_in(self.hui, -1.3, 0.0, 0.55)
        self.camLens.setFilmOffset(0.17, 0.0)          # l'aereo resta al centro dello spazio libero a destra del pannello
        for t in (self.t_info, self.t_top, self.t_nav, self.t_warn, self.t_msg):
            t.hide()
        self.map_root.hide()
        if self.hud_np:
            self.hud_np.hide()
        self.shadow.hide()
        self.rebuild_in_hangar()

    def leave_hangar(self):
        self.h_text()                      # applica anche il testo ancora in modifica
        self.in_hangar = False
        TEX_K[0] = 2
        self.camLens.setFilmOffset(0.0, 0.0)
        self.hui.hide()
        self.hangar_np.hide()
        for t in (self.t_info, self.t_top, self.t_nav, self.t_warn, self.t_msg):
            t.show()
        self.map_root.show()
        if self.hud_np:
            self.hud_np.show()
        for sp in SPECS:
            _TEX_CACHE.pop(sp.short, None)
        _TEX_CACHE.pop('cessna', None)
        _TEX_CACHE.pop('concorde', None)
        self.new_aircraft(self.sel_spec)
        self.ac.reset(False, APTS[self.sel_apt])
        self.show_menu(True)

    def hangar_step(self, dt):
        """Aereo fermo sul pavimento della sala; la camera gli gira attorno."""
        ac, sp = self.ac, self.ac.sp
        ac.fwd, ac.right, ac.up = Vec3(0, 1, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)
        ac.vel = Vec3(0, 0, 0)
        ac.gear_pos = 0.0
        ac.throttle = 0.0
        ac.pos = self.HANGAR_POS + Vec3(0, 0, -sp.wheels[0][1].z - sp.stat * 0.7)
        turn = (1 if self.key('arrow_right') else 0) - (1 if self.key('arrow_left') else 0)
        zoom = (1 if self.key('arrow_down') else 0) - (1 if self.key('arrow_up') else 0)
        self.h_cam += dt * (0.22 + 1.6 * turn)
        self.h_zoom = clamp(self.h_zoom + zoom * dt * 0.8, 0.5, 1.8)

    # ---- minimappa
    def build_minimap(self):
        n = GRID_N + 1
        size = 128
        img = PNMImage(size, size)
        for py in range(size):
            for px in range(size):
                i = min(GRID_N, int(px / size * GRID_N))
                j = min(GRID_N, int((size - 1 - py) / size * GRID_N))
                h = self.terrain.H[j * n + i]
                if h <= 0:
                    c = (0.10, 0.30, 0.50)
                elif h < 300:
                    c = (0.30 + h / 1500, 0.55 - h / 1500, 0.22)
                elif h < 700:
                    c = (0.45, 0.42, 0.38)
                else:
                    c = (0.92, 0.92, 0.92)
                img.setXel(px, py, *c)
        tex = Texture()
        tex.load(img)
        tex.setMagfilter(Texture.FTLinear)
        cm = CardMaker('mappa')
        cm.setFrame(-0.24, 0.24, -0.24, 0.24)
        self.map_root = self.a2dBottomRight.attachNewNode('mappa_root')
        self.map_root.setPos(-0.29, 0, 0.29)
        card = self.map_root.attachNewNode(cm.generate())
        card.setTexture(tex)
        self.map_root.setColorScale(1, 1, 1, 0.92)
        # linee di avvicinamento: tratteggio grigio sottile lungo l'asse di ogni pista (da dove atterrare)
        k = 0.24 / MAP_R
        ls = LineSegs()
        ls.setThickness(1.0)
        ls.setColor(0.72, 0.72, 0.74, 1.0)
        for a in APTS:
            d = a.fwd()
            half = a.L / 2 + 250.0
            for sgn in (1, -1):
                s = half
                while s < half + 16000.0:
                    p0 = (a.cx + d.x * sgn * s, a.cy + d.y * sgn * s)
                    p1 = (a.cx + d.x * sgn * (s + 700.0), a.cy + d.y * sgn * (s + 700.0))
                    ls.moveTo(p0[0] * k, 0, p0[1] * k)
                    ls.drawTo(p1[0] * k, 0, p1[1] * k)
                    s += 1400.0
        self.map_root.attachNewNode(ls.create())

    def draw_minimap(self):
        if self.map_np:
            self.map_np.removeNode()
        ls = LineSegs()
        ls.setThickness(1.5)
        k = 0.24 / MAP_R

        def sq(x, y, r, col):
            ls.setColor(*col)
            ls.moveTo(x - r, 0, y - r)
            for dx, dy in ((r, -r), (r, r), (-r, r), (-r, -r)):
                ls.drawTo(x + dx, 0, y + dy)

        for i, a in enumerate(APTS):
            sq(a.cx * k, a.cy * k, 0.012, (1, 0.9, 0.2, 1) if i == self.dest else (1, 1, 1, 1))
            d = a.fwd()                                             # la pista vera, piu spessa
            ls.setColor(1, 1, 1, 1)
            ls.moveTo((a.cx - d.x * a.L / 2) * k, 0, (a.cy - d.y * a.L / 2) * k)
            ls.drawTo((a.cx + d.x * a.L / 2) * k, 0, (a.cy + d.y * a.L / 2) * k)
        sq(CITY_X * k, CITY_Y * k, 0.007, (0.4, 0.9, 1.0, 1))       # metropoli con il Burj Khalifa
        for rp in self.remotes.values():                            # altri giocatori
            if rp.ac:
                sq(rp.ac.pos.x * k, rp.ac.pos.y * k, 0.006, (0.3, 1.0, 0.4, 1))
        ac = self.ac
        px, py = ac.pos.x * k, ac.pos.y * k
        fx, fy = ac.fwd.x, ac.fwd.y
        ln = math.hypot(fx, fy) or 1.0
        fx, fy = fx / ln, fy / ln
        ls.setColor(1, 0.2, 0.2, 1)
        ls.moveTo(px + fx * 0.02, 0, py + fy * 0.02)
        ls.drawTo(px - fx * 0.012 - fy * 0.012, 0, py - fy * 0.012 + fx * 0.012)
        ls.drawTo(px - fx * 0.012 + fy * 0.012, 0, py - fy * 0.012 - fx * 0.012)
        ls.drawTo(px + fx * 0.02, 0, py + fy * 0.02)
        a = APTS[self.dest]
        ls.setColor(1, 0.9, 0.2, 0.7)
        ls.moveTo(px, 0, py)
        ls.drawTo(a.cx * k, 0, a.cy * k)
        self.map_np = self.map_root.attachNewNode(ls.create())

    # ---- HUD
    def build_hud(self):
        try:
            font = self.loader.loadFont('cmtt12')
        except Exception:
            font = None
        kw = dict(fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.9), mayChange=True)
        if font:
            kw['font'] = font
        self.t_info = OnscreenText(parent=self.a2dBottomLeft, pos=(0.05, 0.46), scale=0.052,
                                   align=TextNode.ALeft, **kw)
        self.t_top = OnscreenText(parent=self.a2dTopCenter, pos=(0, -0.09), scale=0.055,
                                  align=TextNode.ACenter, **kw)
        self.t_nav = OnscreenText(parent=self.a2dTopCenter, pos=(0, -0.16), scale=0.045,
                                  align=TextNode.ACenter, **kw)
        self.t_help = OnscreenText(parent=self.a2dTopRight, pos=(-0.04, -0.17), scale=0.04,
                                   align=TextNode.ARight, text=HELP, **kw)
        self.t_help.hide()
        kw['fg'] = (1, 0.25, 0.2, 1)
        self.t_warn = OnscreenText(parent=self.aspect2d, pos=(0, 0.45), scale=0.085, align=TextNode.ACenter, **kw)
        kw['fg'] = (1, 1, 1, 1)
        self.t_msg = OnscreenText(parent=self.aspect2d, pos=(0, 0.30), scale=0.055, align=TextNode.ACenter, **kw)
        kw['fg'] = (0.6, 1, 0.8, 1)
        self.t_mp = OnscreenText(parent=self.a2dTopLeft, pos=(0.05, -0.09), scale=0.042, align=TextNode.ALeft, **kw)

    def draw_hud_lines(self):
        if self.hud_np:
            self.hud_np.removeNode()
        ac = self.ac
        ls = LineSegs()
        ls.setThickness(2)
        ls.setColor(0.25, 1.0, 0.35, 0.9)
        pitch = math.degrees(math.asin(clamp(ac.fwd.z, -1, 1)))
        roll = math.atan2(-ac.right.z, ac.up.z)
        cr, sr = math.cos(roll), math.sin(roll)
        k = 0.0125

        def pt(x, y):
            return (x * cr - y * sr, x * sr + y * cr)

        for p in range(-90, 91, 5):
            dp = p - pitch
            if abs(dp) > 26:
                continue
            y = dp * k
            w = 0.30 if p == 0 else 0.13 if p % 10 == 0 else 0.06
            g = 0.05
            for a, b in ((-w, -g), (g, w)):
                ls.moveTo(*pt(a, y), 0)
                ls.drawTo(*pt(b, y), 0)
            if p % 10 == 0 and p != 0:
                s = -0.02 if p > 0 else 0.02
                for xx in (-w, w):
                    ls.moveTo(*pt(xx, y), 0)
                    ls.drawTo(*pt(xx, y + s), 0)
        ls.moveTo(-0.02, 0, 0)
        ls.drawTo(0.02, 0, 0)
        ls.moveTo(0, -0.02, 0)
        ls.drawTo(0, 0.02, 0)
        if ac.vel.length() > 6.0:
            fx = clamp(math.degrees(ac.beta) * k, -0.5, 0.5)
            fy = clamp(-math.degrees(ac.alpha) * k, -0.4, 0.4)
            r = 0.022
            for i in range(17):
                a = 2 * math.pi * i / 16
                (ls.moveTo if i == 0 else ls.drawTo)(fx + r * math.cos(a), fy + r * math.sin(a), 0)
            for a, b in (((-0.055, 0), (-r, 0)), ((r, 0), (0.055, 0)), ((0, r), (0, 0.045))):
                ls.moveTo(fx + a[0], fy + a[1], 0)
                ls.drawTo(fx + b[0], fy + b[1], 0)
        self.hud_np = self.aspect2d.attachNewNode(ls.create())

    def update_hud(self):
        ac = self.ac
        ias = ac.vel.length()
        agl = ac.agl()
        hdg = math.degrees(math.atan2(ac.fwd.x, ac.fwd.y)) % 360
        flags = ('  FRENI' if self.ctl.brake else '') + ('  RETRO' if self.ctl.reverse else '')
        if ac.gear_pos < 0.02:
            gear = 'GIU'
        elif ac.gear_pos > 0.98:
            gear = 'SU'
        else:
            gear = 'IN MOTO'
        self.t_info.setText(
            'VEL   %4d km/h  (%d kt)\nQUOTA %5d m  (AGL %d)\nV/S   %+5.1f m/s\nGAS   %3d %%   MOTORI %s\nFLAP  %d/3   CARRELLO %s%s\n%s' % (
                ias * 3.6, ias * 1.944, ac.pos.z, agl, ac.vel.z, ac.throttle * 100,
                ('ACCESI' if ac.spool > 0.98 else 'IN AVVIO') if ac.eng else ('SPENTI' if ac.spool < 0.02 else 'SI SPENGONO'),
                ac.flaps, gear, flags,
                ('QUOTA FISSA %d m (R per sbloccare)' % self.alt_target) if self.alt_hold else 'mouse: libero'))
        self.t_top.setText('%s   RTA %03d   [%s]%s' % (ac.sp.short, hdg, CAM_NAMES[self.cam_mode],
                                                       '   PAUSA' if self.paused else ''))
        a = APTS[self.dest]
        dx, dy = a.cx - ac.pos.x, a.cy - ac.pos.y
        dist = math.hypot(dx, dy)
        brg = math.degrees(math.atan2(dx, dy)) % 360
        eta = dist / max(ias, 20.0) / 60.0
        self.t_nav.setText('DEST %s %s   %.1f km   rotta %03d   ETA %.0f min   ORA %02d:%02d' % (
            a.code, a.name, dist / 1000, brg, eta, int(self.tod), int((self.tod % 1.0) * 60)))
        warn = ''
        vs = ac.stall_speed()
        if ac.crashed:
            warn = 'INCIDENTE'
        elif ac.ditched:
            warn = ''
        elif not ac.eng and not ac.on_ground and ias > 20 and int(self.sim_time * 2) % 2 == 0:
            warn = 'MOTORI SPENTI: premi E'
        elif self.terrain_alert and int(self.sim_time * 3) % 2 == 0:
            warn = 'TERRENO! TIRA SU'
        elif ac.stalled and not ac.on_ground and ias > 8 and int(self.sim_time * 4) % 2 == 0:
            warn = 'STALLO!'
        elif not ac.gear and ac.agl() < 120 and ac.throttle < 0.3 and ias > 20:
            warn = 'CARRELLO!'
        elif ias > ac.sp.vmo:
            warn = 'VELOCITA ECCESSIVA'
        elif not ac.on_ground and ias < 1.3 * vs and ac.throttle < 0.9 and int(self.sim_time * 2) % 2 == 0:
            warn = 'POCA VELOCITA: W'
        self.t_warn.setText(warn)
        if self.in_menu:
            self.t_msg.setText('')
        elif ac.crashed:
            self.t_msg.setText('%s.\nPremi Backspace per ripartire dalla pista.' % ac.crash_reason.capitalize())
        elif ac.ditched:
            self.t_msg.setText(('AMMARAGGIO: l\'aereo e a galla.\n' + ('Scivoli di evacuazione aperti.\n' if ac.ditch_t > 3.0
                                                                       and ac.sp.geo else '')) +
                               'Premi Backspace (pista) o N (in volo) per ripartire.')
        elif self.gate_msg:
            self.t_msg.setText(self.gate_msg)
        elif not ac.eng and ac.spool < 0.02 and ac.on_ground:
            self.t_msg.setText('Motori spenti. Premi E per avviarli.')
        elif ac.eng and ac.spool < 0.98:
            self.t_msg.setText('Avviamento motori...')
        else:
            self.t_msg.setText('')
        self.draw_hud_lines()
        self.draw_minimap()

    # ---- menu
    def build_menu(self):
        """Menu moderno: pannelli scuri a sinistra (aerei) e a destra (aeroporti), centro libero per vedere l'aereo che gira."""
        class _Group:
            def __init__(s, *n):
                s.n = n

            def show(s):
                for x in s.n:
                    x.show()

            def hide(s):
                for x in s.n:
                    x.hide()

            def isHidden(s):
                return s.n[0].isHidden()
        PAN = (0.03, 0.05, 0.09, 0.82)
        ACC = (1, 0.78, 0.25, 1)
        self.m_l = DirectFrame(parent=self.a2dLeftCenter, frameColor=PAN, frameSize=(0, 0.95, -1, 1))
        self.m_r = DirectFrame(parent=self.a2dRightCenter, frameColor=PAN, frameSize=(-0.95, 0, -1, 1))
        self.m_t = DirectFrame(parent=self.a2dTopCenter, frameColor=(0.03, 0.05, 0.09, 0.70), frameSize=(-3, 3, -0.2, 0))
        self.m_b = DirectFrame(parent=self.a2dBottomCenter, frameColor=(0.03, 0.05, 0.09, 0.70), frameSize=(-3, 3, 0, 0.2))
        self.menu = _Group(self.m_l, self.m_r, self.m_t, self.m_b)
        for fr in (self.m_l, self.m_r, self.m_t, self.m_b):
            fr.setTransparency(TransparencyAttrib.MAlpha)
        # filetti gialli come le segnaletiche aeroportuali
        DirectFrame(parent=self.m_l, frameColor=ACC, frameSize=(0.944, 0.95, -1, 1))
        DirectFrame(parent=self.m_r, frameColor=ACC, frameSize=(-0.95, -0.944, -1, 1))
        # striscia gialla e nera sotto la barra del titolo + linea tratteggiata della pista sopra la barra in basso
        self.m_haz = DirectFrame(parent=self.m_t, frameColor=(0, 0, 0, 0), frameSize=(0, 0, 0, 0))
        for k in range(-32, 33):
            DirectFrame(parent=self.m_haz, frameColor=ACC if k % 2 else (0.06, 0.07, 0.10, 1),
                        frameSize=(k * 0.1, k * 0.1 + 0.1, -0.2, -0.186))
        self.m_dash = DirectFrame(parent=self.m_b, frameColor=(0, 0, 0, 0), frameSize=(0, 0, 0, 0))
        for k in range(-16, 17):
            DirectFrame(parent=self.m_dash, frameColor=(1, 1, 1, 0.85), frameSize=(k * 0.2, k * 0.2 + 0.09, 0.190, 0.198))
        OnscreenText(parent=self.m_t, text='FLIGHTWORLD', pos=(0, -0.125), scale=0.10, fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.6))
        OnscreenText(parent=self.m_t, text='AUTORIZZATO AL DECOLLO', pos=(0, -0.172), scale=0.026, fg=ACC)
        self.m_beacon = DirectFrame(parent=self.m_t, frameColor=(1, 0.15, 0.1, 1), frameSize=(-0.62, -0.598, -0.098, -0.076))
        self.m_beacon2 = DirectFrame(parent=self.m_t, frameColor=(0.1, 1, 0.3, 1), frameSize=(0.598, 0.62, -0.098, -0.076))
        for fx, tx, ty, lab in ((self.m_l, 0.05, 0.86, 'AEREO'), (self.m_r, -0.90, 0.86, 'AEROPORTO')):
            wd = 0.036 * len(lab) + 0.05
            DirectFrame(parent=fx, frameColor=ACC, frameSize=(tx - 0.015, tx - 0.015 + wd, ty, ty + 0.078))
            OnscreenText(parent=fx, text=lab, pos=(tx + 0.005, ty + 0.022), scale=0.050, fg=(0.05, 0.06, 0.09, 1), align=TextNode.ALeft)
        acc, dim = ACC, (0.62, 0.70, 0.82, 1)
        self.btn_spec, self.btn_apt = [], []
        for i, sp in enumerate(SPECS):
            self.btn_spec.append(self._mbtn(sp.name, (0.475, 0, 0.78 - 0.100 * i), (-0.425, 0.425),
                                            lambda i=i: self.pick_spec(i), parent=self.m_l, left=True))
        for i, a in enumerate(APTS):
            self.btn_apt.append(self._mbtn('%s  %s' % (a.code, a.name), (-0.475, 0, 0.78 - 0.083 * i), (-0.425, 0.425),
                                           lambda i=i: self.pick_apt(i), parent=self.m_r, left=True, sc=0.040))
        self.t_desc = OnscreenText(parent=self.m_l, text='', pos=(0.05, -0.20), scale=0.038, fg=(0.88, 0.93, 1, 1),
                                   align=TextNode.ALeft, mayChange=True)
        self.t_apt_desc = OnscreenText(parent=self.m_r, text='', pos=(-0.90, -0.40), scale=0.040, fg=(0.88, 0.93, 1, 1),
                                       align=TextNode.ALeft, mayChange=True)
        self.m_extra = [self._mbtn('MULTIPLAYER', (0.475, 0, -0.74), (-0.425, 0.425), self.show_mp, (0.10, 0.36, 0.48, 1), parent=self.m_l),
                        self._mbtn('HANGAR: personalizza', (0.475, 0, -0.87), (-0.425, 0.425), self.enter_hangar, (0.50, 0.33, 0.08, 1), parent=self.m_l)]
        self.btn_bot = []
        for txt, x, cmd, col in (('DAL GATE', -0.62, lambda: self.start('gate'), (0.13, 0.52, 0.24, 1)),
                                 ('DALLA PISTA', -0.31, lambda: self.start(False), (0.13, 0.52, 0.24, 1)),
                                 ('IN VOLO', 0.0, lambda: self.start(True), (0.13, 0.52, 0.24, 1)),
                                 ('RIPRENDI', 0.31, lambda: self.show_menu(False), (0.20, 0.32, 0.55, 1)),
                                 ('ESCI', 0.62, self.userExit, (0.55, 0.16, 0.16, 1))):
            self.btn_bot.append(self._mbtn(txt, (x, 0, 0.075), (-0.145, 0.145), cmd, col, parent=self.m_b, sc=0.040))
        self.ui_hover(self.m_l)
        self.ui_hover(self.m_r)
        self.ui_hover(self.m_b)
        self.taskMgr.add(self._menu_anim, 'menu_anim')
        self.menu.hide()
        self.refresh_menu()

    # ---- schermo intero (pulsante in alto a destra oppure Ctrl+B)
    def build_fs_button(self):
        b = DirectButton(parent=self.a2dTopRight, pos=(-0.085, 0, -0.085), frameSize=(-0.055, 0.055, -0.055, 0.055),
                         frameColor=(0.03, 0.05, 0.09, 0.85), relief=DGG.FLAT, pressEffect=0, command=self.toggle_fullscreen)
        b.setTransparency(TransparencyAttrib.MAlpha)
        acc, a, L, w = (1, 0.78, 0.25, 1), 0.034, 0.026, 0.006
        for sx in (-1, 1):                       # quattro angoli: il simbolo classico di schermo intero
            for sz in (-1, 1):
                DirectFrame(parent=b, frameColor=acc, suppressMouse=0,
                            frameSize=tuple(sorted((sx * a, sx * (a - L)))) + tuple(sorted((sz * a, sz * (a - w)))))
                DirectFrame(parent=b, frameColor=acc, suppressMouse=0,
                            frameSize=tuple(sorted((sx * a, sx * (a - w)))) + tuple(sorted((sz * a, sz * (a - L)))))
        self.fs_btn = b
        self.ui_hover(b)
        self.fs_on = False
        self.fs_win = (1280, 720)

    def toggle_fullscreen(self):
        if ARGS.selftest:
            return
        try:
            wp = WindowProperties()
            if not self.fs_on:
                cur = self.win.getProperties()
                self.fs_win = (cur.getXSize(), cur.getYSize())
                wp.setFullscreen(True)
                wp.setSize(self.pipe.getDisplayWidth(), self.pipe.getDisplayHeight())
            else:
                wp.setFullscreen(False)
                wp.setSize(*self.fs_win)
            self.win.requestProperties(wp)
            self.fs_on = not self.fs_on
        except Exception as ex:
            print('schermo intero non disponibile:', ex)

    # ---- animazioni dell'interfaccia
    def ui_in(self, node, dx=0.0, dy=0.0, dur=0.5, delay=0.0, fade=True, s0=1.0):
        """Fa entrare un elemento dell'interfaccia: scivola da (dx, dy), si dissolve e (opzionale) cresce."""
        home = self._ui_home.setdefault(node, (node.getPos(), node.getScale()))
        old = self._ui_iv.get(node)
        if old:
            old.pause()
        node.setPos(home[0] + Vec3(dx, 0, dy))
        node.setScale(home[1] * s0)
        node.setTransparency(TransparencyAttrib.MAlpha)
        if fade:
            node.setColorScale(1, 1, 1, 0)
        iv = Sequence(Wait(delay), Parallel(
            LerpPosInterval(node, dur, home[0], blendType='easeOut'),
            LerpScaleInterval(node, dur, home[1], blendType='easeOut'),
            LerpColorScaleInterval(node, dur * 0.8, (1, 1, 1, 1))))
        self._ui_iv[node] = iv
        self.clock.tick()                 # dopo un lavoro lungo il tempo dei frame e vecchio: riallinealo
        iv.start()

    def ui_hover(self, root):
        """Effetto al passaggio del mouse su tutti i pulsanti di un pannello: ingrandisce e illumina."""
        for np_ in root.findAllMatches('**/+PGButton'):
            self.accept(np_.node().getEnterEvent(), self._hv, [np_, True])
            self.accept(np_.node().getExitEvent(), self._hv, [np_, False])

    def _hv(self, np_, on, *a):
        old = self._hv_iv.get(np_)
        if old:
            old.pause()
        home = self._hv_home.setdefault(np_, np_.getScale())
        cs = (1.28, 1.28, 1.36, 1) if on else (1, 1, 1, 1)
        iv = Parallel(LerpScaleInterval(np_, 0.10, home * (1.05 if on else 1.0)),
                      LerpColorScaleInterval(np_, 0.10, cs))
        self._hv_iv[np_] = iv
        iv.start()

    def menu_intro(self):
        """Entrata del menu principale: pannelli dai lati, barre da sopra e da sotto, pulsanti a cascata."""
        self.ui_in(self.m_l, -1.1, 0.0, 0.6)
        self.ui_in(self.m_r, 1.1, 0.0, 0.6)
        self.ui_in(self.m_t, 0.0, 0.3, 0.6)
        self.ui_in(self.m_b, 0.0, -0.3, 0.6)
        for i, b in enumerate(self.btn_spec + self.m_extra):
            self.ui_in(b, -0.25, 0.0, 0.4, 0.25 + 0.045 * i)
        for i, b in enumerate(self.btn_apt):
            self.ui_in(b, 0.25, 0.0, 0.4, 0.25 + 0.04 * i)
        for i, b in enumerate(self.btn_bot):
            self.ui_in(b, 0.0, -0.12, 0.4, 0.45 + 0.07 * i)

    def _menu_anim(self, task):
        t = task.time
        if not self.menu.isHidden():
            self.m_dash.setX((t * 0.35) % 0.2)
            self.m_haz.setX(-(t * 0.12) % 0.2)
            self.m_beacon['frameColor'] = (1, 0.15, 0.1, 1 if int(t * 1.6) % 2 == 0 else 0.18)
            self.m_beacon2['frameColor'] = (0.1, 1, 0.3, 1 if int(t * 1.6) % 2 == 1 else 0.18)
        return task.cont

    def _mbtn(self, text, pos, xr, cmd, col=(0.15, 0.17, 0.25, 0.95), parent=None, left=False, sc=0.046):
        kw = dict(text_align=TextNode.ALeft, text_pos=(xr[0] + 0.03, -0.012)) if left else dict(text_pos=(0, -0.012))
        b = DirectButton(parent=parent or self.m_l, text=text, text_scale=sc, text_fg=(1, 1, 1, 1), pos=pos,
                         frameSize=(xr[0], xr[1], -0.043, 0.050), frameColor=col, relief=DGG.FLAT,
                         command=cmd, pressEffect=0, **kw)
        b.setTransparency(TransparencyAttrib.MAlpha)
        b.bar = DirectFrame(parent=b, frameColor=(1, 0.78, 0.25, 1), frameSize=(xr[0], xr[0] + 0.014, -0.043, 0.050),
                            suppressMouse=0)
        b.bar.hide()
        return b

    def refresh_menu(self):
        for lst, sel in ((self.btn_spec, self.sel_spec), (self.btn_apt, self.sel_apt)):
            for i, b in enumerate(lst):
                b['frameColor'] = (0.12, 0.40, 0.85, 1) if i == sel else (0.10, 0.13, 0.21, 0.95)
                b.bar.show() if i == sel else b.bar.hide()
        sp = SPECS[self.sel_spec]
        self.t_desc.setText(sp.desc + '\n\nMassa %d kg   Vr %d km/h' % (sp.mass, sp.vr * 3.6 * 1.1))
        a = APTS[self.sel_apt]
        self.t_apt_desc.setText('%s (%s)\n%d piste, principale %s/%s\n%d m x %d m   quota %d m\n\n%s' % (
            a.name, a.code, len(a.rws), a.rws[0]['labs'][0], a.rws[0]['labs'][1], a.L, a.W, a.h, a.note))

    def pick_spec(self, i):
        self.sel_spec = i
        self.new_aircraft(i)
        self.ac.reset(False, APTS[self.sel_apt])
        self.refresh_menu()

    def pick_apt(self, i):
        self.sel_apt = i
        self.ac.reset(False, APTS[i])
        self.dest = (i + 1) % len(APTS)
        self.refresh_menu()

    def start(self, air):
        if air == 'gate':
            self.ac.reset(False, APTS[self.sel_apt], gate=True)
        else:
            self.ac.reset(air, APTS[self.sel_apt])
        self.dest = (self.sel_apt + 1) % len(APTS)
        self.alt_hold = False
        self.ap_i = 0.0
        self.show_menu(False)

    def show_menu(self, on):
        self.in_menu = on
        if on:
            self.menu.show()
            self.menu_intro()
            self.t_help.hide()
        else:
            self.menu.hide()
        for t in (self.t_info, self.t_top, self.t_nav, self.t_warn, self.t_msg):      # nel menu l'aereo si vede pulito
            t.hide() if on else t.show()
        self.map_root.hide() if on else self.map_root.show()
        if self.hud_np:
            self.hud_np.hide() if on else self.hud_np.show()
        self.refresh_menu()

    def toggle_menu(self):
        if self.in_hangar:
            self.leave_hangar()
        elif not self.mp_ui.isHidden():
            self.hide_mp()
        else:
            self.show_menu(not self.in_menu)

    def destroy(self):
        self.net.stop()
        ShowBase.destroy(self)

    def new_aircraft(self, i):
        self.fx.clear(None)
        if self.ac:
            self.ac.model.removeNode()
        self.ac = Aircraft(self.terrain, SPECS[i])
        self.ac.model.reparentTo(self.render)
        if getattr(self, 'shadow', None):
            self.shadow.removeNode()
        self.shadow = self.build_shadow(SPECS[i])
        self.ctl = Controls()
        self.ap_i = 0.0
        self.alt_hold = False

    def build_shadow(self, sp):
        """Sagoma piatta dell'aereo (vista dall'alto) proiettata sul terreno lungo i raggi del sole."""
        if sp.geo:
            yn, yt, rx, rz, nl, tl = sp.geo['fus']
            ymax, ymin, jet = yn, yt - 0.2, True
            half_b = math.sqrt(sp.S * 9.0) / 2
        else:
            ymax, ymin, jet = 2.5, -5.9, False
            half_b = 5.5
        L = ymax - ymin
        fw = 0.34 if jet else 0.30
        m = Mesh()
        col = (0, 0, 0, 1)
        delta = bool(sp.geo and sp.geo.get('delta'))
        wr = 0.7 if not jet else (1.4 if delta else max(1.0, 0.045 * L))          # semi-larghezza fusoliera
        prof = [(0.0, 0.0), (0.04, 0.45), (0.12, 0.85), (0.3, 1.0), (0.7, 1.0), (0.88, 0.6), (1.0, 0.12)]
        for (f0, w0), (f1, w1) in zip(prof, prof[1:]):
            y0, y1 = ymax - f0 * L, ymax - f1 * L
            a, b = wr * w0, wr * w1
            m.quad([(-a, y0, 0), (a, y0, 0), (b, y1, 0), (-b, y1, 0)], col, (0, (y0 + y1) / 2, 0))
        yw = ymax - fw * L
        cr, ct = 0.18 * L, 0.06 * L
        sweep = 0.47 * half_b if jet else 0.0
        for s in (-1, 1):
            if delta:                                     # ala a delta ogivale
                for (x0, l0), (x1, l1) in zip(CONC_LE, CONC_LE[1:]):
                    m.quad([(s * x0, l0, 0), (s * x1, l1, 0), (s * x1, CONC_TE, 0), (s * x0, CONC_TE, 0)],
                           col, (s * (x0 + x1) / 2, (l0 + l1) / 2 - 4, 0))
                continue
            m.quad([(0, yw, 0), (s * half_b, yw - sweep, 0), (s * half_b, yw - sweep - ct, 0), (0, yw - cr, 0)],
                   col, (s * half_b / 2, yw - cr / 2, 0))
            ht = half_b * (0.36 if jet else 0.33)
            yt0 = ymin + 0.14 * L
            sw = 0.3 * ht if jet else 0.0
            m.quad([(0, yt0, 0), (s * ht, yt0 - sw, 0), (s * ht, yt0 - sw - 0.05 * L, 0), (0, yt0 - 0.10 * L, 0)],
                   col, (s * ht / 2, yt0 - 0.05 * L, 0))
        node = _add(self.render, m, 'ombra')
        node.setLightOff()
        node.setTwoSided(True)
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setDepthWrite(False)
        node.setDepthOffset(2)
        node.setShaderOff()
        return node

    def update_shadow(self):
        ac = self.ac
        if self.in_hangar or (self.fx.crash_on and self.fx.hidden):
            self.shadow.hide()
            return
        L = Vec3(-self.sun_dir.x, -self.sun_dir.y, -self.sun_dir.z)
        h = ac.terrain.sample(ac.pos.x, ac.pos.y)[0]
        p = Vec3(ac.pos)
        for _ in range(3):                                   # il punto d'ombra dipende dalla quota del suolo
            t = max(0.0, (ac.pos.z - h) / max(-L.z, 0.12))
            p = ac.pos + L * t
            h = ac.terrain.sample(p.x, p.y)[0]
        d = 6.0
        hx = ac.terrain.sample(p.x + d, p.y)[0] - ac.terrain.sample(p.x - d, p.y)[0]
        hy = ac.terrain.sample(p.x, p.y + d)[0] - ac.terrain.sample(p.x, p.y - d)[0]
        n = Vec3(-hx / (2 * d), -hy / (2 * d), 1.0)
        n.normalize()
        f = Vec3(ac.fwd.x, ac.fwd.y, 0.0)
        if f.length() < 0.05:
            f = Vec3(ac.up.x, ac.up.y, 0.0)
        f.normalize()
        f = f - n * f.dot(n)
        pos = Vec3(p.x, p.y, h + 0.12)
        self.shadow.setPos(pos)
        self.shadow.lookAt(Point3(pos + f), n)
        agl = max(0.0, ac.pos.z - h)
        a = 0.55 * clamp(1.0 - agl / 2500.0, 0.0, 1.0) * self.shadow_k
        self.shadow.setColorScale(1, 1, 1, a)
        self.shadow.show() if a > 0.02 else self.shadow.hide()

    # ---- comandi
    def next_camera(self):
        if self.in_menu:
            return
        self.cam_mode = (self.cam_mode + 1) % 3
        if self.ac.crashed and self.cam_mode == 1:
            self.cam_mode = 2
        self.ac.set_cockpit(False)
        self.camLens.setFov((70, 80, SPECS[self.sel_spec].tower_fov)[self.cam_mode])

    def toggle_gear(self):
        if not self.in_menu and not self.ac.on_ground:   # a terra il carrello non si ritrae
            self.ac.gear = not self.ac.gear

    def cycle_flaps(self):
        if not self.in_menu:
            self.ac.flaps = (self.ac.flaps + 1) % 4

    def toggle_pause(self):
        self.paused = not self.paused

    def toggle_help(self):
        if self.in_hangar:
            return
        if self.t_help.isHidden():
            self.t_help.show()
            self.ui_in(self.t_help, 0.12, 0.0, 0.35)
        else:
            self.t_help.hide()

    def toggle_althold(self):
        if self.in_menu:
            return
        ac = self.ac
        if ac.crashed or ac.ditched:
            return
        self.alt_hold = not self.alt_hold
        # blocca la quota a cui arrivi tra pochi secondi (tiene conto di salita/discesa in corso),
        # a qualunque altezza, senza mai chiedere di salire per "stare alla larga" dal suolo
        gh = self.terrain.sample(ac.pos.x, ac.pos.y)[0]
        tgt = ac.pos.z + ac.vel.z * abs(ac.vel.z) / 8.0
        self.alt_target = max(tgt, min(ac.pos.z, gh + 8.0))
        self.ap_i = 0.0

    def next_dest(self):
        if not self.in_menu:
            self.dest = (self.dest + 1) % len(APTS)

    def restart_air(self):
        if self.in_menu:
            return
        self.ac.reset(True, self.ac.apt)
        self.alt_hold = False

    def restart_ground(self):
        if self.in_menu:
            return
        self.ac.reset(False, self.ac.apt)
        self.alt_hold = False

    def restart_gate(self):
        if self.in_menu:
            return
        self.ac.reset(False, self.ac.apt, gate=True)
        self.alt_hold = False

    def toggle_engines(self):
        if self.in_menu or self.ac.crashed or self.ac.ditched:
            return
        self.ac.toggle_engines()

    def key(self, name):
        mw = self.mouseWatcherNode
        if mw is None:
            return False
        btn = self.reg.findButton(name)
        return btn is not None and mw.isButtonDown(btn)

    def read_input(self, dt):
        k = self.key
        c = self.ctl
        ac = self.ac
        mw = self.mouseWatcherNode
        if mw is not None and mw.hasMouse():
            cur = (mw.getMouseX(), mw.getMouseY())
            if self.look:
                # col tasto destro premuto il mouse guida solo la visuale: i comandi restano com'erano
                self.lk_ty = (cur[0] - self.look_ref[0]) * -150.0
                self.lk_tp = (cur[1] - self.look_ref[1]) * -60.0
            else:
                self.mouse = cur
        c.brake = k('s') or k('space')
        c.reverse = k('p')
        self.rud_key = (1 if k('d') else 0) - (1 if k('a') else 0)
        if k('w') and ac.eng:
            ac.throttle = min(1.0, ac.throttle + 0.4 * dt)
        if k('s') or k('control'):
            ac.throttle = max(0.0, ac.throttle - 0.5 * dt)

    def flight_controls(self, dt):
        """Il mouse indica dove vuoi andare: converto in comandi di volo."""
        ac, c, sp = self.ac, self.ctl, self.ac.sp
        mx, my = self.fake_mouse if ARGS.selftest else self.mouse
        mx = 0.0 if abs(mx) < 0.05 else mx
        my = 0.0 if abs(my) < 0.05 else my
        V = ac.vel.length()
        rud_key = self.rud_key
        if ac.on_ground or ac.agl() < 1.0:
            tgt_e = clamp(my, -1, 1) if V > 0.75 * sp.vr else 0.0
            pitch_now = math.asin(clamp(ac.fwd.z, -1, 1))
            if pitch_now > ac.max_ground_pitch - 0.03:      # a terra non si fa strisciare la coda
                tgt_e = min(tgt_e, -3.0 * (pitch_now - ac.max_ground_pitch + 0.03) - 0.5 * ac.w[0])
            tgt_a = 0.0
            tgt_r = clamp(mx + rud_key, -1, 1)
            self.ap_i = 0.0
        else:
            roll = math.atan2(-ac.right.z, ac.up.z)
            gamma = math.asin(clamp(ac.vel.z / max(V, 1.0), -1, 1))
            # virate piu strette: l'inclinazione massima si raggiunge a ~75% della corsa del mouse
            bank_des = clamp(mx * 1.35, -1.0, 1.0) * sp.max_bank
            tgt_a = clamp(2.2 * (bank_des - roll) - 1.4 * ac.w[1] * (1.0 if sp.kind == 'prop' else 2.0), -1, 1)
            if self.alt_hold:
                # quota fissa (R): il mouse su/giu viene ignorato, si puo solo virare.
                # Velocita verticale richiesta proporzionale all'errore, poi angolo di rampa: niente picchiata
                vs_cmd = clamp((self.alt_target - ac.pos.z) * 0.12, -0.5 * sp.max_desc * max(V, 20.0),
                               0.8 * sp.max_gamma * max(V, 20.0))
                gdes = clamp(vs_cmd / max(V, 20.0), -sp.max_desc, sp.max_gamma)
            else:
                # proporzionale: ~0 vicino al centro, massimo ai bordi dello schermo
                # (in discesa fino a max_desc: ripida, ma non una picchiata)
                gdes = math.copysign(abs(my) ** 1.2, my) * (sp.max_gamma if my > 0 else sp.max_desc)
            if gdes > 0.0:
                # salita: mai piu ripida di quella che la spinta puo sostenere, cosi non si "siede" e non oscilla;
                # e finche si chiede di salire il gas va su da solo
                gdes = min(gdes, ac.climb_limit())
                if gdes > 0.03 and ac.eng:
                    ac.throttle = min(1.0, ac.throttle + 0.3 * dt)
            # solo un AVVISO se si e in rotta di collisione con il terreno: l'aereo non si tira su da solo,
            # le manovre restano sempre al pilota (vicino agli aeroporti l'avviso e disattivato)
            self.terrain_alert = False
            if not near_airport(ac.pos.x, ac.pos.y, 2000.0):
                for t in (2.0, 4.0, 6.0, 8.0, 10.0):
                    px, py = ac.pos.x + ac.vel.x * t, ac.pos.y + ac.vel.y * t
                    if ac.pos.z + min(ac.vel.z, 0.0) * t - self.terrain.sample(px, py)[0] < 40.0 and ac.vel.z < -2.0:
                        self.terrain_alert = True
                        break
            # protezioni: mai sotto 1.25 x velocita di stallo (se manca spinta il muso scende per riprenderla)
            vmin = 1.25 * ac.stall_speed()
            if V < vmin:
                gdes = min(gdes, (V - vmin) * 0.01)
            # velocita eccessiva: motori al minimo e aerofreni, poi si puo scendere lo stesso
            ac.airbrake = V > 0.90 * sp.vmo and gdes < 0.05
            if ac.airbrake:
                ac.throttle = max(0.1, ac.throttle - 0.3 * dt)
            if V > 1.10 * sp.vmo:
                gdes = max(gdes, (V - 1.10 * sp.vmo) * 0.01)
            a_s = sp.a_s - 0.03 * FLAPS[ac.flaps]
            lim = 0.8 * a_s
            if ac.alpha > lim:
                gdes = min(gdes, gamma) - (ac.alpha - lim) * 2.5
            e = gdes - gamma
            self.ap_i = clamp(self.ap_i + e * 0.4 * dt, -0.5, 0.5)
            ff = ac.trim_elev(math.cos(roll))
            tgt_e = clamp(ff + 3.0 * e + self.ap_i - 1.2 * ac.w[0], -1, 1)
            # virata coordinata: il timone di coda segue l'inclinazione (togliere la sbandata), piu i tasti Q/E
            qr = min(2.5, (V / sp.ref) ** 2 * math.exp(-ac.pos.z / 8500.0))
            qd = clamp(qr, 0.3, 2.5)
            turn_rud = 2.5 * (qd / max(qr, 0.05)) * (G * math.sin(roll) / max(V, 20.0)) / sp.Nr
            tgt_r = clamp(rud_key + turn_rud, -1, 1)
        for attr, tgt, rate in (('elev', tgt_e, 3.0), ('ail', tgt_a, 3.0), ('rud', tgt_r, 3.0)):
            cur = getattr(c, attr)
            setattr(c, attr, cur + clamp(tgt - cur, -rate * dt, rate * dt))

    def skip_hour(self):
        if not self.in_menu and not self.in_hangar:
            self.tod = (self.tod + 1.0) % 24.0

    def look_down(self):
        mw = self.mouseWatcherNode
        if self.in_menu or self.in_hangar or mw is None or not mw.hasMouse():
            return
        self.look = True
        self.look_ref = (mw.getMouseX(), mw.getMouseY())

    def look_up(self):
        self.look = False
        self.lk_ty = self.lk_tp = 0.0

    def wheel(self, d):
        if self.in_hangar:
            self.h_zoom = clamp(self.h_zoom * (1.0 + 0.1 * d), 0.4, 2.5)
        elif not self.in_menu:
            self.cam_zoom = clamp(self.cam_zoom * (1.0 + 0.1 * d), 0.55, 4.0)

    def update_camera(self, dt):
        k = 1 - math.exp(-dt * (10.0 if self.look else 6.0))
        self.lk_y += (self.lk_ty - self.lk_y) * k
        self.lk_p += (self.lk_tp - self.lk_p) * k
        ac = self.ac
        sp = ac.sp
        cam = self.camera
        if getattr(self, 'dbg_cam', None):            # solo per le prove automatiche
            cam.setPos(*self.dbg_cam[0])
            cam.lookAt(Point3(*self.dbg_cam[1]))
            return
        if self.in_hangar:
            ac.set_cockpit(False)
            d = clamp(sp.cam[0] * 0.9, 22.0, 110.0) * self.h_zoom
            hp = self.HANGAR_POS
            cam.setPos(ac.pos + Vec3(math.cos(self.h_cam) * d, math.sin(self.h_cam) * d, d * 0.22 + 2.0))
            cam.lookAt(Point3(ac.pos.x, ac.pos.y, hp.z + sp.cam[1] * 0.28))
            return
        if self.fx.crash_on and self.cam_mode != 2:
            # dopo lo schianto la camera si stacca e gira attorno ai rottami
            ac.set_cockpit(False)
            d = max(sp.cam[0] * 2.0, 60.0 * self.fx.k)
            ang = self.fx.orbit0 + self.fx.t * 0.16
            tgt = ac.pos + Vec3(math.cos(ang) * d, math.sin(ang) * d, d * 0.30 + 4.0)
            gh = self.terrain.sample(tgt.x, tgt.y)[0]
            tgt.z = max(tgt.z, gh + 3.0)
            cur = Vec3(cam.getPos())
            cam.setPos(cur + (tgt - cur) * (1 - math.exp(-dt * 2.5)) + self.fx.shake())
            cam.lookAt(Point3(ac.pos.x, ac.pos.y, ac.pos.z + 3.0 * self.fx.k))
            return
        if self.in_menu:
            ang = self.sim_time * 0.35
            d = sp.cam[0] * 1.75
            cam.setPos(ac.pos + Vec3(math.cos(ang) * d, math.sin(ang) * d, sp.cam[1] * 0.9 + 3.0))
            cam.lookAt(ac.pos + Vec3(0, 0, sp.cam[1] * 0.15))
            gh = self.terrain.sample(cam.getX(), cam.getY())[0]
            if cam.getZ() < gh + 2.0:
                cam.setZ(gh + 2.0)
            ac.set_cockpit(False)
            return
        if self.cam_mode == 1:
            ac.set_cockpit(True)
            cam.setPos(ac.pos + ac.fwd * sp.eye.y + ac.up * sp.eye.z)
            cam.setQuat(ac.quat())
            if abs(self.lk_y) > 0.05 or abs(self.lk_p) > 0.05:
                cam.setHpr(cam, clamp(self.lk_y, -170, 170) * -1.0, clamp(self.lk_p, -80, 80) * 0.7, 0)
            return
        ac.set_cockpit(False)
        if self.cam_mode == 2:
            a = ac.apt
            x, y = a.world(320, 560)
            cam.setPos(x, y, a.h + 45)
            cam.lookAt(ac.pos.x, ac.pos.y, ac.pos.z)
            return
        self.cam_f += (ac.fwd - self.cam_f) * (1 - math.exp(-dt * 4.0))
        self.cam_f.normalize()
        tu = ac.up * 0.5 + Vec3(0, 0, 1) * 0.5
        self.cam_u += (tu - self.cam_u) * (1 - math.exp(-dt * 2.5))
        self.cam_u.normalize()
        z = self.cam_zoom
        off = -self.cam_f * (sp.cam[0] * z) + self.cam_u * (sp.cam[1] * z)
        if abs(self.lk_y) > 0.05 or abs(self.lk_p) > 0.05:
            # clic destro: la camera gira attorno all'aereo senza toccare i comandi di volo
            q = Quat()
            q.setFromAxisAngle(self.lk_y, Vec3(0, 0, 1))
            off = q.xform(off)
            side = off.cross(Vec3(0, 0, 1))
            if side.lengthSquared() > 1e-6:
                side.normalize()
                q2 = Quat()
                q2.setFromAxisAngle(-self.lk_p, side)
                o2 = q2.xform(off)
                if abs(o2.z) < off.length() * 0.97:
                    off = o2
        p = ac.pos + off
        gh = self.terrain.sample(p.x, p.y)[0]
        p.z = max(p.z, gh + 3.0)
        cam.setPos(p)
        cam.lookAt(ac.pos + self.cam_f * sp.cam[0] * z * 0.55 * (1.0 - min(1.0, abs(self.lk_y) / 60.0 + abs(self.lk_p) / 30.0)), self.cam_u)

    def update_gates(self, dt):
        """Passerelle d'imbarco: si attaccano da sole alla porta dell'aereo fermo a motori spenti."""
        ac = self.ac
        self.gate_msg = ''
        apt = next((a for a in APTS if abs(ac.pos.x - a.cx) < a.hx + 800 and abs(ac.pos.y - a.cy) < a.hy + 800), None)
        if apt is None:
            return
        sp = ac.sp
        cargo = getattr(sp, 'cargo', False)
        bridges = [b for b in self.gates.get(apt.code, ()) if (b.kind == 'cargo') == cargo]
        doors = []
        big = True                                           # la passerella serve ogni aereo
        if not hasattr(ac, 'occ'):
            ac.occ = 0
        if sp.geo is not None:
            yn, yt, rx, rz, nl, tl = sp.geo['fus']
            fo, dz = (yn - nl - 9.0, 0.15 * rz) if cargo else (yn - nl - 1.5, -0.25 * rz)    # cargo: portellone dietro la cabina
        else:                                                # Cessna: portiera sul fianco, dietro al muso
            rx, fo, dz = 1.0, 1.0, 0.0
        for sgn in (-1, 1):
            wp = ac.pos + ac.right * (sgn * rx) + ac.fwd * fo + ac.up * dz
            du, dv = apt.local(wp.x, wp.y)
            doors.append(Vec3(du, dv, max(wp.z - apt.h, 1.0)))
        best, bestD, bd = None, None, 1e9
        near = 1e9
        for b in bridges:
            for D in doors:
                dd = (D - b.P).length()
                near = min(near, dd)
                if D.x < b.P.x - 1.0 and dd < 95.0 and dd < bd:
                    best, bestD, bd = b, D, dd
        still = ac.on_ground and ac.vel.length() < 1.5 and not ac.crashed
        can = big and still and not ac.eng and ac.spool < 0.05 and best is not None
        capn = 260 if cargo else SEATS.get(sp.short, 3 if sp.mass < 3000.0 else (12 if sp.mass < 30000.0 else 120))
        what = 'pacchi' if cargo else 'passeggeri'
        for b in bridges:
            tgt = bestD if (can and b is best) else None
            if tgt is None and not b.moving and not b.pax and b.boarded == 0.0 and b.mode == 'off' and (b.tip - b.rest).length() < 0.05:
                continue
            b.update(dt, tgt, capn, ac)
        if big and still and near < 130.0:
            if ac.eng or ac.spool > 0.05:
                self.gate_msg = 'Spegni i motori (E) per collegare ' + ('il nastro cargo' if cargo else 'la passerella')
            elif best is not None:
                if best.docked:
                    if best.mode == 'out':
                        self.gate_msg = 'SBARCO IN CORSO   %d %s a bordo' % (ac.occ, what)
                    elif best.boarded >= best.cap:
                        self.gate_msg = 'CARICO COMPLETATO (%d %s). Avvia i motori (E) per partire' % (best.cap, what)
                    else:
                        self.gate_msg = '%s COLLEGATO   CARICO %d / %d' % ('NASTRO' if cargo else 'PASSERELLA', best.boarded, best.cap)
                else:
                    self.gate_msg = 'Il nastro si sta collegando...' if cargo else 'La passerella si sta collegando...'
            else:
                self.gate_msg = ('Avanza o arretra fino alla linea gialla del dock cargo' if cargo
                                 else 'Avanza o arretra fino alla linea gialla del gate')

    def tick(self, dt):
        """Un passo di simulazione (comandi + fisica), senza grafica."""
        if not ARGS.selftest:
            self.read_input(dt)
        self.flight_controls(dt)
        self.sim_time += dt
        self.acc += dt
        while self.acc >= 1 / 120.0:
            self.ac.step(1 / 120.0, self.ctl)
            self.acc -= 1 / 120.0

    def update(self, task):
        dt = 1 / 60.0 if ARGS.selftest else min(self.clock.getDt(), 0.05)
        if ARGS.selftest and not ARGS.scene:
            self.script()
        if self.in_hangar:
            self.sim_time += dt
            self.hangar_step(dt)
        elif self.in_menu:
            self.sim_time += dt
        elif not self.paused:
            self.tick(dt)
            self.after_tick(dt)
            self.update_gates(dt)
        self.ac.update_visual(dt, self.ctl)
        if not self.paused:
            self.fx.update(dt)
        if not self.paused and not self.in_hangar:
            self.tod = (self.tod + dt / 60.0) % 24.0          # 1 ora di gioco = 1 minuto reale
        self.apply_env(dt)
        self.update_shadow()
        self.update_camera(dt)
        self.sky.setPos(self.camera.getPos())
        self.update_glow(self.sim_time)
        if not self.in_hangar:
            self.update_hud()
        self.audio.update(dt)
        self.mp_update(dt)
        if self.beacon_np is not None:
            on = int(self.sim_time * 1.3) % 2 == 0
            self.beacon_np.setColorScale(1, 1, 1, 1) if on else self.beacon_np.setColorScale(0.12, 0.12, 0.12, 1)
        return task.cont

    def after_tick(self, dt):
        """Eventi dopo la fisica: schianto, tocco pista (fumo gomme), scintille di strisciata."""
        ac, fx = self.ac, self.fx
        if ac.crashed and not fx.crash_on:
            if self.cam_mode == 1:
                self.cam_mode = 0
                self.camLens.setFov(70)
            fx.start_crash(ac)
            self.audio.crash()
        elif not ac.crashed and fx.crash_on:
            fx.clear(ac)
        if not ac.crashed:
            if ac.on_ground and not self.was_ground and self.prev_vz < -0.3 and ac.vel.length() > 25:
                fx.tire_smoke(ac)
                self.audio.one('touch', clamp(0.5 - self.prev_vz * 0.12, 0.4, 1.0))
            if ac.scrape > 0.05 and ac.vel.length() > 8:
                fx.scrape_sparks(ac)
        self.was_ground = ac.on_ground
        self.prev_vz = ac.vel.z

    # ---- prova automatica
    def script(self):
        ac, t = self.ac, self.sim_time
        sp = ac.sp
        if not ac.eng:
            ac.eng, ac.spool = True, 1.0          # la prova automatica parte con i motori accesi
        ac.throttle = 1.0 if t < 120 else 0.6
        ias = ac.vel.length()
        self.fake_mouse = (0.0, 0.6) if ias > sp.vr else (0.0, 0.0)
        if ARGS.air:
            ac.gear = ac.agl() < 30
        if not ac.on_ground and t > 40 and not ARGS.air:
            ac.gear = False                     # ritrae il carrello appena in volo
        if t > 60 and not ac.on_ground:
            self.fake_mouse = (0.3 if int(t / 20) % 2 else -0.3, 0.0 if t < 100 else -0.2)
        if t > 62 and not self.alt_hold and not ac.on_ground:
            self.toggle_althold()               # prova quota fissa: ignora il mouse su/giu

    def telemetry(self, tag=''):
        a = self.ac
        print('%st=%5.1f ias=%6.1f m/s quota=%7.1f vz=%+6.1f pitch=%+5.1f bank=%+5.1f suolo=%s gear=%.2f alpha=%+.2f '
              'gas=%.1f mouse=%s hold=%s tgt=%.0f %s' % (
                  tag, self.sim_time, a.vel.length(), a.pos.z, a.vel.z, math.degrees(math.asin(a.fwd.z)),
                  math.degrees(math.atan2(-a.right.z, a.up.z)), a.on_ground, a.gear_pos, a.alpha, a.throttle,
                  self.fake_mouse, self.alt_hold, self.alt_target, a.crash_reason or ''))

    def place(self, x, y, z, hdg, pitch=0.0, speed=0.0):
        ac = self.ac
        h, p = math.radians(hdg), math.radians(pitch)
        ac.fwd = Vec3(math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p))
        ac.right = Vec3(math.cos(h), -math.sin(h), 0)
        ac.up = ac.right.cross(ac.fwd)
        ac.pos = Vec3(x, y, z)
        ac.vel = ac.fwd * speed
        ac.w = [0.0, 0.0, 0.0]
        self.cam_f = Vec3(ac.fwd)

    def scene_test(self, name):
        """Prove di singole scene con schermate (offscreen)."""
        global HANGAR_FILE
        os.makedirs(ARGS.out, exist_ok=True)
        HANGAR_FILE = os.path.join(ARGS.out, 'hangar_prova.json')      # le prove non toccano le scelte vere

        def run(sec):
            for _ in range(int(sec * 60)):
                self.taskMgr.step()

        def shot(tag):
            self.taskMgr.step()
            self.win.saveScreenshot(Filename.fromOsSpecific(os.path.join(ARGS.out, tag + '.png')))
            print('foto', tag, 'crash=%s' % self.ac.crashed, self.ac.crash_reason, 'particelle', len(self.fx.P))

        ac = self.ac
        if name.startswith('salita') or name.startswith('discesa'):
            # prova dei comandi: mouse in alto (o in basso) per 150 s, con carrello su (default) o giu (_g)
            up = name.startswith('salita')
            ac.reset(True, APTS[0])
            self.in_menu = False
            ac.gear = name.endswith('_g')
            ac.gear_pos = 0.0 if ac.gear else 1.0
            ac.throttle = ac.sp.air_thr
            ac.pos = Vec3(-14000.0, -9000.0, APTS[0].h + (500.0 if up else 3000.0))     # lontano dagli aeroporti
            self.fake_mouse = (0.0, 0.9 if up else -0.9)
            for i in range(150 * 60):
                self.tick(1 / 60.0)
                if i % 300 == 0:
                    print('t=%5.1f ias=%5.1f quota=%6.0f vz=%+5.1f pitch=%+5.1f gas=%.2f %s' % (
                        self.sim_time, ac.vel.length(), ac.pos.z, ac.vel.z,
                        math.degrees(math.asin(ac.fwd.z)), ac.throttle, 'ALLARME' if self.terrain_alert else ''))
                if ac.crashed:
                    print('SCHIANTO', ac.crash_reason)
                    break
            return
        if name == 'carica':
            ld = LoadingIntro(self)
            for tt in (0.4, 1.8, 3.6, 5.0, 5.9, 6.8, 7.6, 8.4, 9.2, 10.0):
                ld.prog = tt / 10.0
                ld.shown = ld.prog
                ld.set_time(tt)
                shot('carica_%04.1f' % tt)
            ld.destroy()
            return
        if name == 'ostacoli':
            seen = set()
            for lst in OBS_GRID.values():
                for o in lst:
                    seen.add(o)
            for a in APTS:
                bad = []
                for (ox, oy, r, top) in seen:
                    u0, v0 = a.local(ox, oy)
                    for r in a.rws:
                        u, v = a.rw_xy(r, u0, v0)
                        if abs(u) < 1800 and abs(v) > r['L'] / 2 + 50 and abs(v) < r['L'] / 2 + 12000 and top - a.h > 25:
                            bad.append((round(u), round(v), round(top - a.h)))
                print(a.code, a.hdg, 'ostacoli nei corridoi:', len(bad), sorted(bad, key=lambda q: abs(q[1]))[:4])
            return
        if name == 'aero':
            # vista dall'alto di ogni aeroporto e un primo piano dei numeri di pista
            self.in_menu = False
            for a in APTS:
                cx, cy = a.world(-300.0 if a.rws[-1]['u'] < 0 else 400.0, 0.0)
                self.dbg_cam = ((cx, cy - 40.0, a.h + 8200.0), (cx, cy, a.h))
                run(0.4)
                shot('aero_' + a.code)
                cx, cy = a.world(-60.0, 700.0)
                self.dbg_cam = ((cx, cy - 20.0, a.h + 2300.0), (cx, cy, a.h))
                run(0.4)
                shot('piantina_' + a.code)
            a = APTS[ARGS.airport]
            r = a.rws[0]
            px, py = a.world(0.0, -r['L'] / 2 + 30.0)
            d = a.fwd()
            self.dbg_cam = ((px - d.x * 90, py - d.y * 90, a.h + 40.0), (px + d.x * 140, py + d.y * 140, a.h))
            run(0.4)
            shot('numero_' + a.code)
            self.dbg_cam = ((px + d.x * 49, py + d.y * 49, a.h + 30.0), (px + d.x * 52, py + d.y * 52, a.h))
            run(0.4)
            shot('numero_alto_' + a.code)
            return
        if name == 'scontro':
            a = Aircraft(self.terrain, SPECS[1])
            b = Aircraft(self.terrain, SPECS[1])
            for dxy, nm in (((0, 60), 'in fila a 60 m'), ((0, 41), 'muso contro coda a 41 m'), ((0, 38.6), 'muso quasi sulla coda'),
                            ((12, 0), 'ala contro fusoliera'), ((30, 0), 'affiancati a 30 m'), ((36, 0), 'ali quasi a contatto'),
                            ((50, 0), 'affiancati a 50 m')):
                for o in (a, b):
                    o.pos, o.fwd, o.right, o.up = Vec3(0, 0, 500), Vec3(0, 1, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)
                b.pos = Vec3(dxy[0], dxy[1], 500)
                print('%-28s scontro=%s' % (nm, planes_collide(a, b)))
            return
        if name == 'menu':
            self.show_menu(True)
            for _k in range(3):
                self.taskMgr.step()
            shot('menu_a')
            run(1.8)
            shot('menu')
            self.show_mp()
            run(1.0)
            shot('mp')
            return
        if name == 'motori':
            ac.reset(False, APTS[0])
            self.in_menu = False
            ac.throttle = 1.0
            for _ in range(5 * 60):
                self.tick(1 / 60.0)
            print('motori spenti, gas 100%%: vel=%.2f spool=%.2f' % (ac.vel.length(), ac.spool))
            ac.toggle_engines()
            for i in range(30 * 60):
                self.tick(1 / 60.0)
                if i % 300 == 0:
                    print('  t=%d spool=%.2f vel=%.1f' % (i // 60, ac.spool, ac.vel.length()))
            ac.toggle_engines()
            ac.throttle = 0.5
            for i in range(20 * 60):
                self.tick(1 / 60.0)
            print('spenti di nuovo: spool=%.2f gas=%.2f vel=%.1f' % (ac.spool, ac.throttle, ac.vel.length()))
            return
        if name == 'gate':
            apt = APTS[ARGS.airport]
            ac.reset(False, apt, gate=True)
            self.in_menu = False
            for t in (0.5, 6.0, 20.0):
                for _ in range(int(t * 60)):
                    self.tick(1 / 60.0)
                    self.update_gates(1 / 60.0)
                    self.after_tick(1 / 60.0)
                print('t=%.1f msg=%r' % (self.sim_time, self.gate_msg), 'eng', ac.eng, 'spool %.2f' % ac.spool)
            b = max(self.gates[apt.code], key=lambda q: q.boarded)
            print('bridge docked', b.docked, 'imbarco %.0f/%d' % (b.boarded, b.cap), 'pax', len(b.pax))
            gv = CARGO_V[0] if ac.sp.cargo else GATE_V[3]
            bx, by = apt.world(GATE_U - 45, gv - 38)
            lx, ly = apt.world(GATE_U - 14, gv)
            for k, (cp, lk) in enumerate((((bx, by, apt.h + 14), (lx, ly, apt.h + 3.5)),
                                          (apt.world(GATE_U - 20, gv + 20) + (apt.h + 7,), (lx, ly, apt.h + 3.2)),
                                          (apt.world(GATE_U - 120, gv - 120) + (apt.h + 40,), (lx, ly, apt.h + 5)))):
                self.dbg_cam = (cp, lk)
                run(0.3)
                shot('gate_%d' % k)
            # ora avvio i motori: la passerella deve ritirarsi
            ac.toggle_engines()
            for _ in range(8 * 60):
                self.tick(1 / 60.0)
                self.update_gates(1 / 60.0)
            print('dopo avvio motori: tip-rest = %.2f' % (b.tip - b.rest).length(), 'msg', repr(self.gate_msg), 'a bordo', ac.occ)
            ac.toggle_engines()                          # spengo di nuovo e riattacco: deve scaricare, poi ricaricare
            for k in range(8):
                for _ in range(5 * 60):
                    self.tick(1 / 60.0)
                    self.update_gates(1 / 60.0)
                print('  t=%.0f modo=%s a bordo=%d pax_in_tubo=%d msg=%r' % (self.sim_time, b.mode, ac.occ, len(b.pax), self.gate_msg))
                if k == 1:
                    shot('scarico')
            return
        if name == 'ore':
            # lo stesso punto a varie ore del giorno (di default sulla pista in volo basso)
            ac.reset(ARGS.air, APTS[ARGS.airport])
            self.in_menu = False
            if ARGS.air:
                ac.eng, ac.spool, ac.throttle = True, 1.0, 0.5
            else:
                ac.eng, ac.spool = True, 1.0
            for hh in (7.0, 12.0, 17.6, 18.4, 19.2, 22.0, 3.0, 5.6):
                self.tod = hh
                self.env_t = 99.0
                for _ in range(30):
                    self.tick(1 / 60.0)
                    self.taskMgr.step()
                self.tod = hh
                run(0.2)
                shot('ora_%04.1f' % hh)
                print('ora', hh, 'sole %.1f' % self.sun_elev(), 'notte %.2f' % self.night)
            self.tod = 22.0
            a0 = APTS[ARGS.airport]
            gx, gy = a0.world(GATE_U - 60.0, GATE_V[3] - 30.0)
            lx, ly = a0.world(GATE_U, GATE_V[3])
            ax, ay = a0.world(-900.0, -a0.L / 2 - 500.0)
            for tag, cp, lk in (('n_citta', (CITY_X - 1700, CITY_Y - 1500, 380.0), (CITY_X, CITY_Y, 250.0)),
                                ('n_aero', (ax, ay, a0.h + 300.0), a0.world(150.0, 0.0) + (a0.h,)),
                                ('n_gate', (gx, gy, a0.h + 12.0), (lx, ly, a0.h + 4.0))):
                self.dbg_cam = (cp, lk)
                self.env_t = 99.0
                run(0.4)
                shot(tag)
            return
        if name == 'visuale':
            ac.reset(True, APTS[ARGS.airport])
            self.in_menu = False
            self.cam_mode = 0
            for _ in range(60):
                self.tick(1 / 60.0)
            for tag, ty, tp, z in (('v0', 0, 0, 1.0), ('v_dx', -100, 0, 1.0), ('v_alto', 0, -45, 1.0),
                                   ('v_vicino', 0, 0, 0.55), ('v_lontano', 0, 0, 3.0), ('v_ritorno', 0, 0, 1.0)):
                self.lk_ty, self.lk_tp, self.cam_zoom = ty, tp, z
                self.look = bool(ty or tp)
                for _ in range(90):
                    self.tick(1 / 60.0)
                    self.update_camera(1 / 60.0)
                shot(tag)
                print(tag, 'visuale', round(self.lk_y), round(self.lk_p), 'zoom', z, 'mouse comandi', self.mouse)
            self.cam_mode = 1
            self.lk_ty, self.lk_tp = 90, 0
            for _ in range(90):
                self.tick(1 / 60.0)
                self.update_camera(1 / 60.0)
            shot('v_cabina')
            return
        if name == 'gate2':
            # parcheggio "a mano": fermo sul piazzale lontano dalla linea gialla, con motori spenti
            apt = APTS[ARGS.airport]
            for du, dv, dh in ((-25.0, 15.0, 0.0), (-35.0, -30.0, 35.0), (-50.0, 0.0, -90.0), (-12.0, 40.0, 180.0)):
                ac.reset(False, apt, gate=True)
                self.in_menu = False
                x0, y0, z0 = ac.pos.x, ac.pos.y, ac.pos.z
                hd = math.degrees(math.atan2(ac.fwd.x, ac.fwd.y)) + dh
                dx, dy = apt.world(GATE_U - 17.0 + du, GATE_V[3] + dv)
                self.place(dx, dy, z0, hd)
                for _ in range(20 * 60):
                    self.tick(1 / 60.0)
                    self.update_gates(1 / 60.0)
                    self.after_tick(1 / 60.0)
                b = max(self.gates[apt.code], key=lambda q: q.boarded)
                print('du=%+.0f dv=%+.0f dh=%+.0f ->' % (du, dv, dh), 'vel %.2f' % ac.vel.length(), 'suolo', ac.on_ground,
                      'agganciata', b.docked, 'imbarco %.0f/%d' % (b.boarded, b.cap), repr(self.gate_msg))
            return
        if name == 'livree':
            self.enter_hangar()
            run(0.3)
            for i, lvi in ((5, 0), (5, 4), (1, 3), (3, 2), (2, 11), (4, 3), (1, 1), (6, 12), (7, 13), (8, 14), (3, 8), (2, 4)):
                self.h_pick_ac(i)
                c = CUSTOM.setdefault(SPECS[i].short, {})
                c['livery'] = lvi
                if lvi:
                    c['name'] = LIVERIES[lvi]['name']
                apply_custom(SPECS[i])
                self.h_cam = 0.9
                self.rebuild_in_hangar()
                run(0.4)
                shot('livrea_%d_%d' % (i, lvi))
                self.h_cam = 2.45
                self.hui.hide()
                run(0.4)
                shot('coda_%d_%d' % (i, lvi))
                self.h_cam = -0.45
                self.h_zoom = 1.5
                run(0.4)
                shot('codaD_%d_%d' % (i, lvi))
                self.h_zoom = 1.0
                self.hui.show()
            return
        if name == 'probe':
            def corridor(a):
                worst = 0.0
                for sgn in (1, -1):
                    for d in range(600, 15000, 300):
                        for lat in (-900, 0, 900):
                            u, v = lat, sgn * (a.L / 2 + d)
                            x, y = a.world(u, v)
                            h = max(terrain_raw(x, y), 0.0)
                            worst = max(worst, (h - a.h) / d)
                            # citta: grattacieli entro 2500 m dal centro (alti fino a ~830 m)
                            dc = math.hypot(x - CITY_X, y - CITY_Y)
                            if dc < 2600:
                                worst = max(worst, (150.0 - 0.0) / d)
                return worst
            for a in APTS:
                res = []
                for hd in ('N', 'E'):
                    b = Airport(a.code, a.name, a.cx, a.cy, a.h, hd, a.L)
                    res.append('%s=%.3f' % (hd, corridor(b)))
                print(a.code, a.cx, a.cy, 'pendenza max corridoi:', ' '.join(res))
            for yy in range(44000, -45000, -2500):
                row = ''
                for xx in range(-44000, 45000, 2000):
                    ch = '.' if terrain_raw(xx, yy) <= 0 else ('#' if terrain_raw(xx, yy) > 600 else '+')
                    for a in APTS:
                        if abs(xx - a.cx) < 1000 and abs(yy - a.cy) < 1250:
                            ch = 'A'
                    row += ch
                print('%6d %s' % (yy, row))
            return
        if name == 'quota':
            # tasto R a varie quote e in salita/discesa: deve tenere la quota senza picchiare
            for agl0, my0 in ((1500, 0.9), (1500, -0.9), (1500, 0.0), (80, 0.0), (80, -0.5), (25, 0.0)):
                ac.reset(True, APTS[0])
                self.in_menu = False
                ac.gear, ac.gear_pos = False, 1.0
                ac.throttle = ac.sp.air_thr
                self.alt_hold = False
                x0, y0 = -14000.0, -9000.0
                gh = self.terrain.sample(x0, y0)[0]
                self.place(x0, y0, gh + agl0, 0, 0.0, ac.sp.vair * 1.4)
                self.fake_mouse = (0.0, my0)
                for i in range(8 * 60):
                    self.tick(1 / 60.0)
                zmin = ac.pos.z
                self.toggle_althold()
                self.fake_mouse = (0.0, 0.0 if my0 == 0 else my0)    # il mouse resta dov'e: R lo ignora
                print('--- agl0=%d mouse=%+.1f  R a z=%.0f vz=%+.1f target=%.0f' % (agl0, my0, ac.pos.z, ac.vel.z, self.alt_target))
                for i in range(40 * 60):
                    self.tick(1 / 60.0)
                    zmin = min(zmin, ac.pos.z)
                    if i % 360 == 0:
                        print('  t=%2d z=%6.0f agl=%5.0f vz=%+5.1f pitch=%+5.1f ias=%5.1f gas=%.2f' % (
                            i // 60, ac.pos.z, ac.agl(), ac.vel.z, math.degrees(math.asin(ac.fwd.z)), ac.vel.length(), ac.throttle))
                    if ac.crashed or ac.ditched:
                        print('  FINE:', ac.crash_reason or 'ammarato')
                        break
                print('  quota minima %.0f  finale %.0f' % (zmin, ac.pos.z))
            return
        if name == 'ala':
            # atterraggio con ala/motore a terra: solo scintille (sulla pista), nessuno schianto
            for bank, vz0 in ((25, -2.0), (40, -3.0)):
                ac.reset(True, APTS[0])
                self.in_menu = False
                ac.gear, ac.gear_pos = True, 0.0
                ac.throttle = 0.0
                x, y = APTS[0].world(0, -APTS[0].L / 2 + 400)
                self.place(x, y, APTS[0].h + 7.0, 0, -3.0, ac.sp.vair * 0.9)
                self.fake_mouse = (0.0, 0.0)
                b = math.radians(bank)
                ac.right, ac.up = rot(ac.right, ac.fwd, b), rot(ac.up, ac.fwd, b)
                ac.vel = Vec3(ac.vel.x, ac.vel.y, vz0)
                sp_seen = False
                for i in range(10 * 60):
                    self.tick(1 / 60.0)
                    sp_seen = sp_seen or ac.scrape > 0
                    if ac.crashed:
                        break
                print('bank=%d: crashed=%s %s scintille=%s quota=%.1f' % (bank, ac.crashed, ac.crash_reason, sp_seen, ac.pos.z))
            return
        if name == 'virata':
            ac.reset(True, APTS[0])
            self.in_menu = False
            ac.gear, ac.gear_pos = False, 1.0
            ac.throttle = ac.sp.air_thr
            self.place(-14000.0, -9000.0, self.terrain.sample(-14000.0, -9000.0)[0] + 1500, 0, 0.0, ac.sp.vair * 1.4)
            self.fake_mouse = (0.8, 0.0)
            h0 = math.atan2(ac.fwd.x, ac.fwd.y)
            for i in range(40 * 60):
                self.tick(1 / 60.0)
                if i % 300 == 299:
                    h = math.atan2(ac.fwd.x, ac.fwd.y)
                    roll = math.degrees(math.atan2(-ac.right.z, ac.up.z))
                    print('t=%2d bank=%+5.1f beta=%+5.2f rud=%+5.2f vz=%+5.1f ias=%5.1f rta=%4.0f' % (
                        i // 60, roll, math.degrees(ac.beta), self.ctl.rud, ac.vel.z, ac.vel.length(), math.degrees(h) % 360))
                if i == 59 or i == 600:
                    self.update_camera(0.1)
                    self.ac.update_visual(1 / 60.0, self.ctl)
                    self.cam_mode = 0
                    for _ in range(20):
                        self.taskMgr.step()
                    self.win.saveScreenshot(Filename.fromOsSpecific(os.path.join(ARGS.out, 'virata_%d.png' % i)))
            return
        if name in ('mp_host', 'mp_join'):
            # due processi: l'host scrive il codice in un file, il client lo legge e si collega (127.0.0.1)
            codefile = os.path.join(os.path.dirname(os.path.abspath(ARGS.out)), 'mp_code.txt')
            self.in_menu = False
            ac.reset(False, APTS[0])
            if name == 'mp_host':
                self.net.start_host('HOST')
                with open(codefile, 'w') as f:
                    f.write(self.net.local_code().replace(self.net.local_ip(), '127.0.0.1'))
            else:
                for _ in range(300):
                    if os.path.exists(codefile):
                        break
                    time.sleep(0.1)
                time.sleep(0.5)
                code = open(codefile).read()
                self.sel_spec = 1
                self.new_aircraft(1)
                self.ac.reset(False, APTS[0])
                ac = self.ac
                ac.pos.y += 120.0
                self.net.start_client(code, 'AMICO')
            for i in range(1500):
                self.taskMgr.step()
                time.sleep(1 / 100.0)
                if name == 'mp_join':
                    ac.throttle, self.fake_mouse = 0.0, (0.0, 0.0)
                if i % 300 == 299:
                    print(name, 't=%d' % i, 'stato:', self.net.status, '| giocatori', self.net.players(),
                          '| remoti', {k: (r.ac.sp.short if r.ac else None) for k, r in self.remotes.items()})
                if i == 1200:
                    self.win.saveScreenshot(Filename.fromOsSpecific(os.path.join(ARGS.out, name + '.png')))
            self.net.stop()
            return
        if name == 'hangar':
            self.enter_hangar()
            run(0.5)
            shot('hangar_a')
            self.h_set_color('accent', self.PALETTE[2])
            self.h_set_color('body', self.PALETTE[7])
            self.h_set_level('power', 2)
            run(2.0)
            shot('hangar_b')
            self.h_pick_ac(3)
            run(1.0)
            shot('hangar_a380')
            self.h_reset()
            self.leave_hangar()
            run(0.2)
            print('uscito dall hangar, in_menu=', self.in_menu, 'T0=', self.ac.sp.T0)
        elif name in ('citta', 'burj'):
            self.paused = True
            self.in_menu = False
            ac.reset(True, APTS[1])
            self.paused = True
            if name == 'citta':
                self.place(CITY_X - 4200, CITY_Y - 4200, 700, 45)
            else:
                self.place(CITY_X - 700, CITY_Y - 700, 450, 45)
            run(0.3)
            shot(name)
            if name == 'burj':
                self.place(CITY_X - 1500, CITY_Y + 100, 420, 90)
                run(0.3)
                shot('burj_lontano')
        else:
            ac.reset(True, APTS[0])
            self.in_menu = False
            self.cam_mode = 0
            ac.gear = False
            ac.gear_pos = 1.0
            ac.throttle = 0.8
            if name == 'schianto':
                x, y = APTS[0].world(-900, 200)
                gh = self.terrain.sample(x, y)[0]
                self.place(x, y, gh + 7, 0, -2.0, ac.sp.vair * 1.4)
            elif name == 'acqua':
                self.place(-46000, 8000, 7.0, 90, -5.0, ac.sp.vair * 1.3)
            elif name == 'edificio':
                self.place(CITY_X - 900, CITY_Y + 10, 260, 90, 0.0, ac.sp.vair * 1.2)
            for i in range(300):
                self.taskMgr.step()
                if ac.crashed:
                    break
            print('schianto avvenuto:', ac.crashed, ac.crash_reason, 'ammarato:', ac.ditched, 'a t=%.1f' % self.sim_time)
            if name == 'acqua':
                run(6.0)
                print('ammarato: z=%.2f vel=%.1f' % (ac.pos.z, ac.vel.length()))
            for tag, sec in (('0', 0.1), ('1', 0.5), ('2', 1.5), ('3', 3.0), ('4', 8.0), ('5', 25.0)):
                run(sec - (self.fx.t if self.fx.crash_on else 0.0))
                shot(name + '_' + tag)

    def selftest(self):
        os.makedirs(ARGS.out, exist_ok=True)
        shots = {60: 'pista', 2580: 'carrello_a_meta', 3600: 'volo', 6000: 'volo2'}
        for f in range(1, 6001):
            if f in (62, 64):          # vista cabina e vista ravvicinata orbitante
                self.cam_mode = 1 if f == 62 else 0
                self.in_menu = f == 64
                for _ in range(3):
                    self.taskMgr.step()
                self.win.saveScreenshot(Filename.fromOsSpecific(
                    os.path.join(ARGS.out, 'cabina.png' if f == 62 else 'orbita.png')))
                self.in_menu = False
                self.cam_mode = 0
            if f in (66, 68, 70):         # tre quarti anteriore, dall'alto-dietro, di fronte
                t0 = self.sim_time
                self.sim_time = {66: 1.1, 68: 4.0, 70: 1.6}[f] / 0.35
                self.in_menu = True
                for _ in range(3):
                    self.taskMgr.step()
                self.win.saveScreenshot(Filename.fromOsSpecific(
                    os.path.join(ARGS.out, {66: 'fronte.png', 68: 'dietro.png', 70: 'muso.png'}[f])))
                self.in_menu = False
                self.sim_time = t0
            if f in shots:
                self.taskMgr.step()
                self.win.saveScreenshot(Filename.fromOsSpecific(os.path.join(ARGS.out, shots[f] + '.png')))
            else:
                self.script()
                self.tick(1 / 60.0)
            if f % 300 == 0:
                self.telemetry()
        m = self.render.getRelativeVector(self.ac.model, Vec3(0, 1, 0))
        print('orientamento modello vs fwd:', m, self.ac.fwd)


if __name__ == '__main__':
    app = FlightSim()
    if ARGS.selftest and ARGS.scene:
        app.scene_test(ARGS.scene)
    elif ARGS.selftest:
        app.selftest()
    else:
        app.run()

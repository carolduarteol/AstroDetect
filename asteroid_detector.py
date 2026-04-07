"""
Asteroid Detector for Astrometrica Software Images
====================================================
Detecta asteroides em imagens FITS usadas pelo Astrometrica,
usando comparação de frames, subtração de fundo e astrometria.

Autor: AstroDetect Python
Requisitos: astropy, numpy, scipy, matplotlib, photutils, sep
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import ZScaleInterval, ImageNormalize
import astropy.units as u
from scipy.ndimage import median_filter, gaussian_filter
from scipy.spatial import KDTree

try:
    import sep
    HAS_SEP = True
except ImportError:
    HAS_SEP = False

try:
    from photutils.detection import DAOStarFinder, IRAFStarFinder
    from photutils.background import Background2D, MedianBackground
    from photutils.aperture import CircularAperture, aperture_photometry
    HAS_PHOTUTILS = True
except ImportError:
    HAS_PHOTUTILS = False

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  CARREGAMENTO DE IMAGENS FITS
# ─────────────────────────────────────────────

class FITSLoader:
    """Carrega e prepara imagens FITS no formato do Astrometrica."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.data = None
        self.header = None
        self.wcs = None
        self._load()

    def _load(self):
        logger.info(f"Carregando FITS: {self.filepath.name}")
        with fits.open(self.filepath) as hdul:
            # Astrometrica geralmente usa a extensão primária
            idx = 0
            for i, hdu in enumerate(hdul):
                if hdu.data is not None:
                    idx = i
                    break
            self.header = hdul[idx].header
            data = hdul[idx].data.astype(np.float64)
            while data.ndim > 2:
                data = data[0]
            self.data = data

        # WCS (coordenadas celestes)
        try:
            self.wcs = WCS(self.header)
            if not self.wcs.has_celestial:
                self.wcs = None
        except Exception:
            self.wcs = None

        logger.info(f"  Dimensões: {self.data.shape} | WCS: {'OK' if self.wcs else 'Ausente'}")

    @property
    def obs_time(self):
        """Retorna hora de observação do header."""
        for kw in ['DATE-OBS', 'DATE_OBS', 'JD', 'MJD-OBS']:
            if kw in self.header:
                return self.header[kw]
        return None

    @property
    def exposure(self):
        for kw in ['EXPTIME', 'EXPOSURE']:
            if kw in self.header:
                return float(self.header[kw])
        return None


# ─────────────────────────────────────────────
#  PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────

class Preprocessor:
    """Subtração de bias/dark/flat e estimativa de fundo."""

    def __init__(self, sigma=3.0, box_size=64):
        self.sigma = sigma
        self.box_size = box_size

    def subtract_background(self, data: np.ndarray):
        """Estima e subtrai o fundo (sky background)."""
        if HAS_PHOTUTILS:
            sigma_clip = SigmaClip(sigma=self.sigma)
            bkg_estimator = MedianBackground()
            try:
                bkg = Background2D(
                    data, self.box_size,
                    filter_size=(3, 3),
                    sigma_clip=sigma_clip,
                    bkg_estimator=bkg_estimator,
                )
                return data - bkg.background, bkg.background_rms_median
            except Exception:
                pass

        # Fallback: sigma-clipped stats
        mean, median, std = sigma_clipped_stats(data, sigma=self.sigma)
        return data - median, std

    def apply_flat_correction(self, data: np.ndarray, flat: np.ndarray):
        """Aplica correção de flat-field."""
        flat_norm = flat / np.median(flat)
        flat_norm[flat_norm < 0.1] = 1.0
        return data / flat_norm

    def apply_dark_correction(self, data: np.ndarray, dark: np.ndarray):
        """Subtrai dark frame."""
        return data - dark

    def remove_cosmics(self, data: np.ndarray, threshold: float = 5.0):
        """Remove raios cósmicos simples via filtro de mediana."""
        med = median_filter(data, size=3)
        diff = np.abs(data - med)
        mad = np.median(diff)
        mask = diff > threshold * mad * 1.4826
        cleaned = data.copy()
        cleaned[mask] = med[mask]
        return cleaned


# ─────────────────────────────────────────────
#  DETECÇÃO DE FONTES
# ─────────────────────────────────────────────

class SourceDetector:
    """Detecta fontes pontuais (estrelas e candidatos a asteroides)."""

    def __init__(self, fwhm=3.0, threshold_sigma=5.0):
        self.fwhm = fwhm
        self.threshold_sigma = threshold_sigma

    def detect(self, data: np.ndarray, bkg_rms: float = None):
        """Retorna tabela de fontes detectadas."""
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        if bkg_rms is None:
            bkg_rms = std

        threshold = self.threshold_sigma * bkg_rms

        if HAS_SEP:
            return self._detect_sep(data, threshold)
        elif HAS_PHOTUTILS:
            return self._detect_photutils(data, mean, threshold)
        else:
            return self._detect_simple(data, threshold)

    def _detect_sep(self, data, threshold):
        """Detecção via SEP (SourceExtractor Python)."""
        data_c = np.ascontiguousarray(data, dtype=np.float64)
        bkg = sep.Background(data_c)
        data_sub = data_c - bkg
        objects = sep.extract(data_sub, threshold, err=bkg.globalrms)
        sources = []
        for obj in objects:
            sources.append({
                'x': float(obj['x']),
                'y': float(obj['y']),
                'flux': float(obj['flux']),
                'peak': float(obj['peak']),
                'a': float(obj['a']),
                'b': float(obj['b']),
                'theta': float(obj['theta']),
                'ellipticity': 1.0 - float(obj['b']) / max(float(obj['a']), 1e-6),
            })
        return sources

    def _detect_photutils(self, data, mean, threshold):
        """Detecção via photutils DAOStarFinder."""
        daofind = DAOStarFinder(fwhm=self.fwhm, threshold=threshold)
        sources_table = daofind(data - mean)
        if sources_table is None:
            return []
        sources = []
        for row in sources_table:
            sources.append({
                'x': float(row['xcentroid']),
                'y': float(row['ycentroid']),
                'flux': float(row['flux']),
                'peak': float(row['peak']),
                'a': self.fwhm / 2.35,
                'b': self.fwhm / 2.35,
                'theta': 0.0,
                'ellipticity': 0.0,
                'sharpness': float(row['sharpness']),
                'roundness': float(row['roundness1']),
            })
        return sources

    def _detect_simple(self, data, threshold):
        """Detecção simples por limiares (fallback sem dependências extras)."""
        from scipy.ndimage import label, center_of_mass, maximum_filter
        smoothed = gaussian_filter(data, sigma=self.fwhm / 2.35)
        local_max = (smoothed == maximum_filter(smoothed, size=int(self.fwhm * 3)))
        above = smoothed > threshold
        mask = local_max & above
        labeled, num = label(mask)
        sources = []
        for i in range(1, num + 1):
            cy, cx = center_of_mass(data, labeled, i)
            peak = data[int(cy), int(cx)] if 0 <= int(cy) < data.shape[0] and 0 <= int(cx) < data.shape[1] else 0
            sources.append({
                'x': float(cx), 'y': float(cy),
                'flux': float(peak),
                'peak': float(peak),
                'a': self.fwhm, 'b': self.fwhm,
                'theta': 0.0, 'ellipticity': 0.0,
            })
        return sources


# ─────────────────────────────────────────────
#  CORRESPONDÊNCIA DE FONTES (MATCHING)
# ─────────────────────────────────────────────

class SourceMatcher:
    """
    Encontra fontes em comum entre frames (estrelas estacionárias)
    e identifica objetos em movimento (candidatos a asteroides).
    """

    def __init__(self, match_radius_px: float = 5.0):
        self.match_radius = match_radius_px

    def match(self, sources_a: list, sources_b: list):
        """
        Retorna:
          matched   - pares (idx_a, idx_b) de fontes coincidentes
          only_a    - fontes presentes só em A (desapareceram)
          only_b    - fontes presentes só em B (apareceram)
        """
        if not sources_a or not sources_b:
            return [], list(range(len(sources_a))), list(range(len(sources_b)))

        coords_a = np.array([[s['x'], s['y']] for s in sources_a])
        coords_b = np.array([[s['x'], s['y']] for s in sources_b])

        tree_b = KDTree(coords_b)
        dists, idxs = tree_b.query(coords_a, k=1)

        matched = []
        used_b = set()
        used_a = set()

        for i, (dist, j) in enumerate(zip(dists, idxs)):
            if dist <= self.match_radius and j not in used_b:
                matched.append((i, int(j)))
                used_a.add(i)
                used_b.add(int(j))

        only_a = [i for i in range(len(sources_a)) if i not in used_a]
        only_b = [j for j in range(len(sources_b)) if j not in used_b]

        return matched, only_a, only_b


# ─────────────────────────────────────────────
#  ALINHAMENTO DE FRAMES
# ─────────────────────────────────────────────

class FrameAligner:
    """Alinha frames consecutivos usando estrelas de referência."""

    def __init__(self, min_stars: int = 6):
        self.min_stars = min_stars

    def compute_shift(self, sources_ref: list, sources_tgt: list,
                      matcher: SourceMatcher):
        """Calcula translação simples (dx, dy) entre dois frames."""
        matched, _, _ = matcher.match(sources_ref, sources_tgt)
        if len(matched) < self.min_stars:
            logger.warning(f"Poucos pares para alinhamento: {len(matched)}")
            return 0.0, 0.0

        shifts = []
        for i, j in matched:
            dx = sources_tgt[j]['x'] - sources_ref[i]['x']
            dy = sources_tgt[j]['y'] - sources_ref[i]['y']
            shifts.append((dx, dy))

        shifts = np.array(shifts)
        # Média robusta
        dx = np.median(shifts[:, 0])
        dy = np.median(shifts[:, 1])
        return dx, dy

    def apply_shift(self, data: np.ndarray, dx: float, dy: float):
        """Aplica translação inteira ao array."""
        from scipy.ndimage import shift
        return shift(data, (-dy, -dx), cval=0)


# ─────────────────────────────────────────────
#  FILTRAGEM DE CANDIDATOS
# ─────────────────────────────────────────────

class AsteroidFilter:
    """
    Filtra candidatos a asteroides com base em:
    - Morfologia (pontual, não muito elíptico)
    - Movimento coerente entre frames
    - SNR mínimo
    """

    def __init__(self,
                 min_snr: float = 5.0,
                 max_ellipticity: float = 0.5,
                 min_motion_px: float = 0.5,
                 max_motion_px: float = 50.0):
        self.min_snr = min_snr
        self.max_ellipticity = max_ellipticity
        self.min_motion = min_motion_px
        self.max_motion = max_motion_px

    def filter_morphology(self, sources: list, bkg_rms: float):
        """Remove fontes muito elípticas (traços, galáxias)."""
        filtered = []
        for s in sources:
            ellip = s.get('ellipticity', 0.0)
            snr = s.get('flux', 0) / max(bkg_rms, 1e-9)
            if ellip <= self.max_ellipticity and snr >= self.min_snr:
                filtered.append(s)
        return filtered

    def find_moving_objects(self, candidates_per_frame: list):
        """
        candidates_per_frame: lista de listas de candidatos por frame.
        Retorna objetos que se movem de forma coerente entre todos os frames.
        """
        if len(candidates_per_frame) < 2:
            return candidates_per_frame[0] if candidates_per_frame else []

        matcher = SourceMatcher(match_radius_px=self.max_motion)
        ref = candidates_per_frame[0]
        moving = []

        for src in ref:
            track = [src]
            found_all = True
            for frame_srcs in candidates_per_frame[1:]:
                # Procura a fonte mais próxima no frame seguinte
                if not frame_srcs:
                    found_all = False
                    break
                coords = np.array([[s['x'], s['y']] for s in frame_srcs])
                dists = np.sqrt((coords[:, 0] - track[-1]['x'])**2 +
                                (coords[:, 1] - track[-1]['y'])**2)
                idx = np.argmin(dists)
                if dists[idx] < self.max_motion:
                    track.append(frame_srcs[idx])
                else:
                    found_all = False
                    break

            if not found_all:
                continue

            # Verifica movimento mínimo entre primeiro e último frame
            total_dx = track[-1]['x'] - track[0]['x']
            total_dy = track[-1]['y'] - track[0]['y']
            total_motion = np.sqrt(total_dx**2 + total_dy**2)

            if self.min_motion <= total_motion <= self.max_motion * len(candidates_per_frame):
                # Verifica linearidade do movimento
                if self._is_linear(track):
                    src_with_track = src.copy()
                    src_with_track['track'] = track
                    src_with_track['motion_px'] = total_motion
                    src_with_track['motion_angle'] = np.degrees(np.arctan2(total_dy, total_dx))
                    moving.append(src_with_track)

        return moving

    def _is_linear(self, track: list, max_deviation: float = 3.0):
        """Verifica se o movimento é aproximadamente linear."""
        if len(track) < 3:
            return True
        xs = np.array([s['x'] for s in track])
        ys = np.array([s['y'] for s in track])
        t = np.arange(len(track))
        # Ajuste linear
        px = np.polyfit(t, xs, 1)
        py = np.polyfit(t, ys, 1)
        res_x = xs - np.polyval(px, t)
        res_y = ys - np.polyval(py, t)
        rms = np.sqrt(np.mean(res_x**2 + res_y**2))
        return rms < max_deviation


# ─────────────────────────────────────────────
#  ASTROMETRIA (PIXEL → RA/DEC)
# ─────────────────────────────────────────────

class Astrometry:
    """Converte coordenadas de pixel para RA/Dec usando WCS."""

    def __init__(self, wcs: WCS = None):
        self.wcs = wcs

    def pixel_to_radec(self, x: float, y: float):
        """Retorna (ra, dec) em graus ou None se WCS indisponível."""
        if self.wcs is None:
            return None, None
        try:
            coord = self.wcs.pixel_to_world(x, y)
            return coord.ra.deg, coord.dec.deg
        except Exception:
            return None, None

    def format_mpc(self, candidates: list, obs_code: str = '500',
                   obs_date: str = None):
        """
        Formata candidatos no formato MPC (Minor Planet Center).
        https://www.minorplanetcenter.net/iau/info/OpticalObs.html
        """
        lines = []
        for i, cand in enumerate(candidates):
            ra, dec = cand.get('ra'), cand.get('dec')
            if ra is None or dec is None:
                continue
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            ra_str = coord.ra.to_string(unit=u.hour, sep=' ', pad=True, precision=2)
            dec_str = coord.dec.to_string(sep=' ', alwayssign=True, pad=True, precision=1)
            mag = cand.get('magnitude', 99.9)
            date = obs_date or datetime.utcnow().strftime('%Y %m %d.%f')[:16]
            line = (f"{'':5s}{'C':1s}{str(date):16s} "
                    f"{ra_str:11s} {dec_str:11s}         "
                    f"{mag:5.1f} R      {obs_code:3s}")
            lines.append(line)
        return '\n'.join(lines)


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

class AsteroidDetectionPipeline:
    """
    Pipeline completo de detecção de asteroides em imagens Astrometrica.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.preprocessor = Preprocessor(
            sigma=cfg.get('bg_sigma', 3.0),
            box_size=cfg.get('bg_box', 64)
        )
        self.detector = SourceDetector(
            fwhm=cfg.get('fwhm', 3.0),
            threshold_sigma=cfg.get('threshold_sigma', 5.0)
        )
        self.matcher = SourceMatcher(
            match_radius_px=cfg.get('match_radius', 5.0)
        )
        self.aligner = FrameAligner(
            min_stars=cfg.get('min_align_stars', 6)
        )
        self.asteroid_filter = AsteroidFilter(
            min_snr=cfg.get('min_snr', 5.0),
            max_ellipticity=cfg.get('max_ellipticity', 0.5),
            min_motion_px=cfg.get('min_motion_px', 0.5),
            max_motion_px=cfg.get('max_motion_px', 50.0)
        )
        self.config = cfg

    def run(self, fits_files: list):
        """
        Executa o pipeline em múltiplos frames FITS.
        Retorna lista de candidatos a asteroides.
        """
        logger.info(f"\n{'='*55}")
        logger.info(f"  ASTEROID DETECTOR — {len(fits_files)} frame(s)")
        logger.info(f"{'='*55}")

        frames = []
        for fpath in fits_files:
            loader = FITSLoader(fpath)
            frames.append(loader)

        # 1) Pré-processamento
        logger.info("\n[1/4] Pré-processamento...")
        processed = []
        bkg_rmss = []
        for frame in frames:
            data_c = self.preprocessor.remove_cosmics(frame.data)
            data_sub, bkg_rms = self.preprocessor.subtract_background(data_c)
            processed.append(data_sub)
            bkg_rmss.append(bkg_rms)
            logger.info(f"  {Path(frame.filepath).name}: BKG_RMS={bkg_rms:.2f}")

        # 2) Detecção de fontes por frame
        logger.info("\n[2/4] Detecção de fontes...")
        all_sources = []
        for i, (data, bkg_rms) in enumerate(zip(processed, bkg_rmss)):
            sources = self.detector.detect(data, bkg_rms)
            sources = self.asteroid_filter.filter_morphology(sources, bkg_rms)
            all_sources.append(sources)
            logger.info(f"  Frame {i+1}: {len(sources)} fontes detectadas")

        # 3) Alinhamento (se múltiplos frames)
        if len(frames) > 1:
            logger.info("\n[2.5/4] Alinhamento de frames...")
            ref_sources = all_sources[0]
            aligned_data = [processed[0]]
            aligned_sources = [all_sources[0]]
            for i in range(1, len(frames)):
                dx, dy = self.aligner.compute_shift(ref_sources, all_sources[i], self.matcher)
                logger.info(f"  Frame {i+1} → dx={dx:.2f}px, dy={dy:.2f}px")
                aligned_d = self.aligner.apply_shift(processed[i], dx, dy)
                # Re-detecta após alinhamento
                _, bkg_rms = self.preprocessor.subtract_background(aligned_d)
                srcs = self.detector.detect(aligned_d, bkg_rms)
                srcs = self.asteroid_filter.filter_morphology(srcs, bkg_rms)
                aligned_data.append(aligned_d)
                aligned_sources.append(srcs)
            processed = aligned_data
            all_sources = aligned_sources

        # 4) Identificar objetos em movimento
        logger.info("\n[3/4] Buscando objetos em movimento...")

        if len(frames) == 1:
            # Frame único: usa fontes não coincidentes com catálogo (modo simples)
            candidates = all_sources[0]
            logger.warning("  Apenas 1 frame: não é possível detectar movimento. "
                           "Retornando todas as fontes como candidatos.")
        else:
            candidates = self.asteroid_filter.find_moving_objects(all_sources)

        logger.info(f"  {len(candidates)} candidato(s) a asteroide encontrado(s)")

        # 5) Astrometria
        logger.info("\n[4/4] Calculando coordenadas celestes...")
        wcs = frames[0].wcs
        astrometry = Astrometry(wcs)

        results = []
        for cand in candidates:
            ra, dec = astrometry.pixel_to_radec(cand['x'], cand['y'])
            result = {
                'x_px': round(cand['x'], 2),
                'y_px': round(cand['y'], 2),
                'ra_deg': round(ra, 6) if ra else None,
                'dec_deg': round(dec, 6) if dec else None,
                'flux': round(cand.get('flux', 0), 2),
                'snr': round(cand.get('flux', 0) / max(bkg_rmss[0], 1), 2),
                'ellipticity': round(cand.get('ellipticity', 0), 3),
                'motion_px': round(cand.get('motion_px', 0), 2),
                'motion_angle_deg': round(cand.get('motion_angle', 0), 1),
                'obs_time': frames[0].obs_time,
                'exposure_s': frames[0].exposure,
            }
            if ra and dec:
                logger.info(f"  Candidato: RA={ra:.4f}° Dec={dec:.4f}° "
                            f"motion={result['motion_px']:.1f}px")
            else:
                logger.info(f"  Candidato: x={result['x_px']} y={result['y_px']} "
                            f"(sem WCS) motion={result['motion_px']:.1f}px")
            results.append(result)

        return {
            'candidates': results,
            'frames': [str(f.filepath) for f in frames],
            'n_frames': len(frames),
            'processed_at': datetime.utcnow().isoformat(),
        }


# ─────────────────────────────────────────────
#  VISUALIZAÇÃO
# ─────────────────────────────────────────────

class Visualizer:
    """Gera imagens de diagnóstico e saída visual."""

    def plot_detections(self, data: np.ndarray, candidates: list,
                        output_path: str = None, title: str = "Asteroid Candidates"):
        """Plota imagem com candidatos marcados."""
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0a0f')
        ax.set_facecolor('#0a0a0f')

        norm = ImageNormalize(data, interval=ZScaleInterval())
        ax.imshow(data, cmap='gray', norm=norm, origin='lower')

        for i, cand in enumerate(candidates):
            x, y = cand['x_px'], cand['y_px']
            circle = patches.Circle(
                (x, y), radius=12, linewidth=1.5,
                edgecolor='#00ff88', facecolor='none', linestyle='--'
            )
            ax.add_patch(circle)
            label = f"#{i+1}"
            if cand.get('motion_px', 0) > 0:
                label += f"\n{cand['motion_px']:.1f}px"
            ax.text(x + 14, y, label, color='#00ff88',
                    fontsize=8, va='center', fontfamily='monospace')

            # Traçar trilha se disponível
            if 'track' in cand:
                track = cand['track']
                xs = [t['x'] for t in track]
                ys = [t['y'] for t in track]
                ax.plot(xs, ys, color='#ff6600', linewidth=1.0,
                        linestyle='-', alpha=0.7, marker='x', markersize=4)

        ax.set_title(title, color='white', fontsize=13, pad=12)
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

        info = f"{len(candidates)} candidato(s) detectado(s)"
        ax.text(0.02, 0.02, info, transform=ax.transAxes,
                color='#aaa', fontsize=9, va='bottom')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                        facecolor='#0a0a0f')
            logger.info(f"  Imagem salva: {output_path}")
        else:
            plt.show()
        plt.close()

    def plot_blink(self, frames_data: list, candidates: list, output_path: str = None):
        """Comparação lado a lado dos frames com candidatos."""
        n = min(len(frames_data), 4)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), facecolor='#0a0a0f')
        if n == 1:
            axes = [axes]

        for i, (ax, data) in enumerate(zip(axes, frames_data[:n])):
            ax.set_facecolor('#0a0a0f')
            norm = ImageNormalize(data, interval=ZScaleInterval())
            ax.imshow(data, cmap='gray', norm=norm, origin='lower')
            ax.set_title(f'Frame {i+1}', color='white', fontsize=10)
            ax.tick_params(colors='gray', labelsize=7)

            for cand in candidates:
                if 'track' in cand and i < len(cand['track']):
                    tx = cand['track'][i]['x']
                    ty = cand['track'][i]['y']
                else:
                    tx, ty = cand['x_px'], cand['y_px']
                circle = patches.Circle(
                    (tx, ty), radius=10, linewidth=1.5,
                    edgecolor='#ff4444', facecolor='none'
                )
                ax.add_patch(circle)

        plt.suptitle("Blink Comparator — Candidatos a Asteroide",
                     color='white', fontsize=12, y=1.01)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                        facecolor='#0a0a0f')
            logger.info(f"  Blink comparator salvo: {output_path}")
        else:
            plt.show()
        plt.close()


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Detector de Asteroides para imagens do Astrometrica (FITS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Três frames consecutivos:
  python asteroid_detector.py frame1.fits frame2.fits frame3.fits

  # Com configuração personalizada e saída de imagem:
  python asteroid_detector.py *.fits --threshold 4.0 --fwhm 3.5 --output resultado.json --plot

  # Frame único (lista todas as fontes):
  python asteroid_detector.py imagem.fits --plot
        """
    )

    parser.add_argument('fits_files', nargs='+', help='Arquivos FITS de entrada')
    parser.add_argument('--fwhm', type=float, default=3.0, help='FWHM estimado em pixels (padrão: 3.0)')
    parser.add_argument('--threshold', type=float, default=5.0, help='Limiar de detecção em sigma (padrão: 5.0)')
    parser.add_argument('--match-radius', type=float, default=5.0, help='Raio de correspondência em px (padrão: 5.0)')
    parser.add_argument('--min-motion', type=float, default=0.5, help='Movimento mínimo em px (padrão: 0.5)')
    parser.add_argument('--max-motion', type=float, default=50.0, help='Movimento máximo por frame em px (padrão: 50.0)')
    parser.add_argument('--min-snr', type=float, default=5.0, help='SNR mínimo (padrão: 5.0)')
    parser.add_argument('--obs-code', default='500', help='Código do observatório MPC (padrão: 500=geocêntrico)')
    parser.add_argument('--output', '-o', help='Salvar resultados em JSON')
    parser.add_argument('--plot', action='store_true', help='Gerar visualizações')
    parser.add_argument('--plot-dir', default='.', help='Diretório para salvar plots')
    parser.add_argument('--mpc', action='store_true', help='Mostrar saída no formato MPC')
    parser.add_argument('--verbose', '-v', action='store_true', help='Log detalhado')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Verificar existência dos arquivos
    for f in args.fits_files:
        if not Path(f).exists():
            logger.error(f"Arquivo não encontrado: {f}")
            sys.exit(1)

    config = {
        'fwhm': args.fwhm,
        'threshold_sigma': args.threshold,
        'match_radius': args.match_radius,
        'min_motion_px': args.min_motion,
        'max_motion_px': args.max_motion,
        'min_snr': args.min_snr,
    }

    pipeline = AsteroidDetectionPipeline(config)
    results = pipeline.run(args.fits_files)

    # Resumo
    candidates = results['candidates']
    print(f"\n{'─'*50}")
    print(f"  RESULTADO: {len(candidates)} candidato(s) encontrado(s)")
    print(f"{'─'*50}")
    for i, c in enumerate(candidates):
        print(f"\n  Candidato #{i+1}")
        print(f"    Posição (px): x={c['x_px']}, y={c['y_px']}")
        if c['ra_deg'] is not None:
            print(f"    RA/Dec:       {c['ra_deg']:.6f}° / {c['dec_deg']:.6f}°")
        print(f"    Fluxo/SNR:    {c['flux']:.1f} / {c['snr']:.1f}")
        if c['motion_px'] > 0:
            print(f"    Movimento:    {c['motion_px']:.2f}px @ {c['motion_angle_deg']:.1f}°")

    # Formato MPC
    if args.mpc and any(c['ra_deg'] for c in candidates):
        print(f"\n{'─'*50}")
        print("  FORMATO MPC")
        print(f"{'─'*50}")
        astrometry = Astrometry()
        # Prepara candidatos para MPC
        mpc_candidates = [
            {'ra': c['ra_deg'], 'dec': c['dec_deg'],
             'magnitude': 99.9 - min(c['snr'] / 2, 10)}
            for c in candidates if c['ra_deg'] is not None
        ]
        obs_time = candidates[0].get('obs_time') if candidates else None
        print(astrometry.format_mpc(mpc_candidates, args.obs_code, obs_time))

    # Salvar JSON
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"\nResultados salvos em: {args.output}")

    # Visualizações
    if args.plot and candidates:
        viz = Visualizer()
        loader = FITSLoader(args.fits_files[0])
        data_sub, _ = pipeline.preprocessor.subtract_background(loader.data)
        while data_sub.ndim > 2:
            data_sub = data_sub[0]
        plot_path = Path(args.plot_dir) / 'asteroid_detections.png'
        # Adapta para usar dict compatível
        plot_cands = []
        for c in candidates:
            pc = dict(c)
            pc['x'] = c['x_px']
            pc['y'] = c['y_px']
            # Reconstrói track se existir
            plot_cands.append(pc)
        viz.plot_detections(data_sub, plot_cands, str(plot_path))

        if len(args.fits_files) > 1:
            frames_data = []
            for fp in args.fits_files[:4]:
                ld = FITSLoader(fp)
                d, _ = pipeline.preprocessor.subtract_background(ld.data)
                frames_data.append(d)
            blink_path = Path(args.plot_dir) / 'asteroid_blink.png'
            viz.plot_blink(frames_data, plot_cands, str(blink_path))

    return results


if __name__ == '__main__':
    main()

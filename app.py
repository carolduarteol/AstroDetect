"""
AstroDetect -- Servidor Flask
=============================
Roda o pipeline de deteccao de asteroides e serve a interface web.

Uso:
  pip install flask
  python app.py
  Abre http://localhost:5000 no navegador
"""

import os
import io
import json
import base64
import logging
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

# Importa o pipeline do asteroid_detector.py (deve estar na mesma pasta)
try:
    from asteroid_detector import (
        AsteroidDetectionPipeline, Visualizer, FITSLoader, Astrometry
    )
    PIPELINE_OK = True
except ImportError as e:
    PIPELINE_OK = False
    PIPELINE_ERROR = str(e)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------
#  HTML DA INTERFACE
# ---------------------------------------------

HTML = '''<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AstroDetect</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#040810; font-family:'Syne',sans-serif; color:#e8f4ff; min-height:100vh; padding:24px; }
  .app { max-width:820px; margin:0 auto; }

  .header { background:#0d1520; border:0.5px solid #1e2d45; border-radius:16px 16px 0 0; padding:18px 24px; display:flex; align-items:center; gap:12px; }
  .logo { width:36px; height:36px; background:#0e4a8a; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:18px; border:1px solid #1a6bbf; }
  .header-title { font-size:20px; font-weight:800; color:#e8f4ff; letter-spacing:-0.3px; }
  .header-sub { font-size:11px; color:#4a7aaa; font-family:'Space Mono',monospace; margin-top:2px; }
  .status { margin-left:auto; display:flex; align-items:center; gap:6px; font-size:11px; font-family:'Space Mono',monospace; color:#2a9d5c; }
  .dot { width:6px; height:6px; border-radius:50%; background:#2a9d5c; animation:pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  .body { background:#080c14; border:0.5px solid #1e2d45; border-top:none; border-radius:0 0 16px 16px; padding:24px; display:flex; flex-direction:column; gap:22px; }

  .sec-title { font-size:11px; color:#2a5a8a; font-family:'Space Mono',monospace; text-transform:uppercase; letter-spacing:.8px; margin-bottom:10px; }

  .dropzone { border:1px dashed #1e3a5f; border-radius:12px; background:#0a1220; padding:32px 24px; text-align:center; cursor:pointer; transition:all .2s; position:relative; }
  .dropzone:hover, .dropzone.over { border-color:#2a6aad; background:#0c1828; }
  .dropzone input { position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; }
  .drop-icon { font-size:30px; margin-bottom:10px; opacity:.5; }
  .drop-title { font-size:15px; font-weight:600; color:#b8d4f0; margin-bottom:4px; }
  .drop-sub { font-size:12px; color:#3a5a7a; font-family:'Space Mono',monospace; }

  .file-list { display:flex; flex-direction:column; gap:6px; margin-top:10px; }
  .file-item { background:#0d1826; border:0.5px solid #1a2d45; border-radius:8px; padding:10px 14px; display:flex; align-items:center; gap:10px; font-family:'Space Mono',monospace; font-size:12px; animation:fi .3s ease; }
  .file-name { flex:1; color:#8abde0; }
  .file-size { color:#2a4a6a; }
  .file-del { cursor:pointer; color:#2a4a6a; transition:color .2s; }
  .file-del:hover { color:#c0392b; }
  @keyframes fi { from{opacity:0;transform:translateY(-4px)} to{opacity:1;transform:translateY(0)} }

  .params { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .param { display:flex; flex-direction:column; gap:6px; }
  .param label { font-size:11px; color:#3a6a9a; font-family:'Space Mono',monospace; text-transform:uppercase; letter-spacing:.5px; }
  .param input { background:#0a1220; border:0.5px solid #1a3050; border-radius:6px; padding:9px 12px; font-family:'Space Mono',monospace; font-size:13px; color:#8abde0; outline:none; transition:border-color .2s; width:100%; }
  .param input:focus { border-color:#1a6bbf; }

  .checks { display:flex; gap:20px; flex-wrap:wrap; }
  .check { display:flex; align-items:center; gap:8px; cursor:pointer; }
  .check input { accent-color:#1a6bbf; width:14px; height:14px; cursor:pointer; }
  .check span { font-size:13px; color:#5a8ab0; }

  .run-btn { background:#0e4a8a; border:none; border-radius:8px; padding:15px 24px; font-family:'Syne',sans-serif; font-size:15px; font-weight:600; color:#e8f4ff; cursor:pointer; transition:all .2s; width:100%; }
  .run-btn:hover { background:#1a6bbf; }
  .run-btn:active { transform:scale(.98); }
  .run-btn:disabled { background:#0a2a3a; color:#2a4a6a; cursor:not-allowed; }

  .progress { height:3px; background:#0a1a2a; border-radius:2px; overflow:hidden; display:none; }
  .progress.show { display:block; }
  .progress-bar { height:100%; background:#1a6bbf; border-radius:2px; transition:width .4s ease; }

  .log-box { background:#04080f; border:0.5px solid #0e1e30; border-radius:8px; padding:14px; font-family:'Space Mono',monospace; font-size:11px; max-height:200px; overflow-y:auto; display:none; line-height:1.9; }
  .log-box.show { display:block; }
  .li { color:#4a9fd0; } .lok { color:#2a9d5c; } .lw { color:#d4a017; } .le { color:#c0392b; } .ld { color:#2a4a6a; }

  .results { display:none; }
  .results.show { display:block; }
  .res-header { font-size:13px; font-weight:600; color:#4a7aaa; font-family:'Space Mono',monospace; text-transform:uppercase; letter-spacing:.5px; margin-bottom:12px; display:flex; align-items:center; gap:8px; }
  .badge { background:#0e3a6a; color:#4a9fd0; border-radius:4px; padding:2px 8px; font-size:11px; }

  .candidate { background:#0a1422; border:0.5px solid #1a2d45; border-left:2px solid #1a6bbf; border-radius:8px; padding:12px 16px; margin-bottom:8px; display:grid; grid-template-columns:48px 1fr auto; gap:12px; align-items:center; animation:fi .4s ease; }
  .cand-num { font-family:'Space Mono',monospace; font-size:11px; color:#2a5a8a; background:#0d1f35; border-radius:4px; padding:4px 8px; text-align:center; }
  .cand-coords { font-family:'Space Mono',monospace; font-size:12px; color:#6aadda; }
  .cand-motion { font-size:11px; color:#3a6a8a; margin-top:3px; }
  .cand-snr { font-family:'Space Mono',monospace; font-size:14px; color:#2a9d5c; text-align:right; }
  .cand-snr-label { font-size:10px; color:#1a5a3a; margin-top:1px; }

  .mpc-box { background:#04080f; border:0.5px solid #0e1e30; border-radius:8px; padding:14px; font-family:'Space Mono',monospace; font-size:11px; color:#2a9d5c; white-space:pre; overflow-x:auto; line-height:2; }
  .mpc-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
  .copy-btn { background:#0d1826; border:0.5px solid #1a3050; border-radius:4px; padding:5px 12px; font-family:'Space Mono',monospace; font-size:11px; color:#3a6a9a; cursor:pointer; transition:all .2s; }
  .copy-btn:hover { border-color:#1a6bbf; color:#4a9fd0; }

  .img-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .img-card { background:#0a1220; border:0.5px solid #1a2d45; border-radius:8px; overflow:hidden; }
  .img-card img { width:100%; display:block; }
  .img-label { font-size:11px; color:#3a6a8a; font-family:'Space Mono',monospace; padding:8px 12px; text-align:center; }

  .error-box { background:#1a0808; border:0.5px solid #4a1515; border-radius:8px; padding:14px; font-family:'Space Mono',monospace; font-size:12px; color:#c0392b; line-height:1.8; display:none; }
  .error-box.show { display:block; }
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="logo">*</div>
    <div>
      <div class="header-title">AstroDetect</div>
      <div class="header-sub">asteroid detection pipeline v1.0 -- flask server</div>
    </div>
    <div class="status"><div class="dot"></div>ONLINE</div>
  </div>

  <div class="body">

    <div>
      <div class="sec-title">Frames FITS</div>
      <div class="dropzone" id="dz">
        <input type="file" id="fi" multiple accept=".fits,.fit" onchange="handleFiles(this.files)" style="position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;">
        <div class="drop-icon">&#10022;</div>
        <div class="drop-title">Arraste os arquivos FITS aqui</div>
        <div class="drop-sub">ou clique para selecionar -- minimo 3 frames</div>
      </div>
      <div class="file-list" id="fl"></div>
    </div>

    <div>
      <div class="sec-title">Parametros</div>
      <div class="params">
        <div class="param"><label>Threshold (sigma)</label><input type="number" id="threshold" value="8.0" step="0.5" min="1"></div>
        <div class="param"><label>FWHM (pixels)</label><input type="number" id="fwhm" value="3.0" step="0.5" min="1"></div>
        <div class="param"><label>Movimento min (px)</label><input type="number" id="minMotion" value="2.0" step="0.5" min="0"></div>
        <div class="param"><label>Movimento max (px)</label><input type="number" id="maxMotion" value="20.0" step="1" min="1"></div>
        <div class="param"><label>SNR minimo</label><input type="number" id="minSnr" value="8.0" step="0.5" min="1"></div>
        <div class="param"><label>Codigo MPC obs.</label><input type="text" id="obsCode" value="500" maxlength="3"></div>
      </div>
    </div>

    <div>
      <div class="sec-title">Opcoes</div>
      <div class="checks">
        <label class="check"><input type="checkbox" id="chkPlot" checked><span>Gerar grafico PNG</span></label>
        <label class="check"><input type="checkbox" id="chkMpc" checked><span>Formato MPC</span></label>
        <label class="check"><input type="checkbox" id="chkBlink" checked><span>Blink comparator</span></label>
      </div>
    </div>

    <div class="progress" id="progress"><div class="progress-bar" id="pb" style="width:0%"></div></div>

    <button class="run-btn" id="runBtn" onclick="runPipeline()">> Executar pipeline</button>

    <div class="log-box" id="log"></div>
    <div class="error-box" id="errBox"></div>

    <div class="results" id="results">

      <div class="res-header">Candidatos detectados <span class="badge" id="cnt">0</span></div>
      <div id="candList"></div>

      <div id="mpcSection" style="display:none; margin-top:16px;">
        <div class="mpc-header">
          <div class="sec-title" style="margin:0">Formato MPC</div>
          <button class="copy-btn" onclick="copyMpc()">copiar</button>
        </div>
        <div class="mpc-box" id="mpcBox"></div>
      </div>

      <div id="imgSection" style="display:none; margin-top:16px;">
        <div class="sec-title">Visualizacoes</div>
        <div class="img-grid" id="imgGrid"></div>
      </div>

    </div>

  </div>
</div>

<script>
  var files = [];

  function handleFiles(f) {
    for (var i=0;i<f.length;i++) files.push(f[i]);
    renderFiles();
  }

  function renderFiles() {
    var el = document.getElementById('fl');
    el.innerHTML = '';
    files.forEach(function(f,i) {
      el.innerHTML += '<div class="file-item">' +
        '<span style="color:#1a6bbf">*</span>' +
        '<span class="file-name">'+f.name+'</span>' +
        '<span class="file-size">'+((f.size/1024/1024).toFixed(1))+'MB</span>' +
        '<span class="file-del" onclick="removeFile('+i+')">X</span>' +
        '</div>';
    });
  }

  function removeFile(i) { files.splice(i,1); renderFiles(); }

  var dz = document.getElementById('dz');
  dz.addEventListener('dragover',  function(e){e.preventDefault();dz.classList.add('over');});
  dz.addEventListener('dragleave', function(){dz.classList.remove('over');});
  dz.addEventListener('drop', function(e){e.preventDefault();dz.classList.remove('over');handleFiles(e.dataTransfer.files);});

  function log(msg,cls) {
    var b=document.getElementById('log');
    b.classList.add('show');
    b.innerHTML+='<div class="'+(cls||'ld')+'">'+msg+'</div>';
    b.scrollTop=b.scrollHeight;
  }

  function setProg(p) {
    document.getElementById('progress').classList.add('show');
    document.getElementById('pb').style.width=p+'%';
  }

  async function runPipeline() {
    if (files.length < 3) { alert('Selecione pelo menos 3 arquivos FITS!'); return; }

    var btn = document.getElementById('runBtn');
    btn.disabled = true; btn.textContent = '... Processando...';
    document.getElementById('log').innerHTML = '';
    document.getElementById('log').classList.remove('show');
    document.getElementById('results').classList.remove('show');
    document.getElementById('errBox').classList.remove('show');
    document.getElementById('errBox').innerHTML = '';
    setProg(10);

    var fd = new FormData();
    files.forEach(function(f){ fd.append('fits_files', f); });
    fd.append('threshold',  document.getElementById('threshold').value);
    fd.append('fwhm',       document.getElementById('fwhm').value);
    fd.append('min_motion', document.getElementById('minMotion').value);
    fd.append('max_motion', document.getElementById('maxMotion').value);
    fd.append('min_snr',    document.getElementById('minSnr').value);
    fd.append('obs_code',   document.getElementById('obsCode').value);
    fd.append('plot',       document.getElementById('chkPlot').checked ? '1' : '0');
    fd.append('mpc',        document.getElementById('chkMpc').checked  ? '1' : '0');
    fd.append('blink',      document.getElementById('chkBlink').checked? '1' : '0');

    log('[INFO] Enviando ' + files.length + ' frames para o servidor...', 'li');
    setProg(20);

    try {
      var resp = await fetch('/run', { method: 'POST', body: fd });
      setProg(80);
      var data = await resp.json();
      setProg(100);

      if (data.error) {
        document.getElementById('errBox').innerHTML = 'ERRO: ' + data.error;
        document.getElementById('errBox').classList.add('show');
        btn.disabled=false; btn.textContent='> Executar pipeline';
        return;
      }

      data.log.forEach(function(l) {
        var cls = l.includes('[INFO]')?'li': l.includes('[WARNING]')?'lw': l.includes('ERROR')?'le':'ld';
        log(l, cls);
      });

      showResults(data);

    } catch(e) {
      document.getElementById('errBox').innerHTML = 'Erro de conexao: ' + e.message;ge;
      document.getElementById('errBox').classList.add('show');
    }

    btn.disabled=false; btn.textContent='> Executar pipeline';
  }

  function showResults(data) {
    var res = document.getElementById('results');
    res.classList.add('show');
    document.getElementById('cnt').textContent = data.candidates.length;

    var list = document.getElementById('candList');
    list.innerHTML = '';
    data.candidates.forEach(function(c,i) {
      var ra  = c.ra_deg  != null ? c.ra_deg.toFixed(6)+'deg'  : 'N/A';
      var dec = c.dec_deg != null ? c.dec_deg.toFixed(6)+'deg' : 'N/A';
      list.innerHTML += '<div class="candidate">' +
        '<div class="cand-num">#'+(i+1)+'</div>' +
        '<div><div class="cand-coords">RA '+ra+'  Dec '+dec+'</div>' +
        '<div class="cand-motion">movimento: '+c.motion_px+'px @ '+c.motion_angle_deg+'deg</div></div>' +
        '<div class="cand-snr">'+c.snr.toFixed(1)+'<div class="cand-snr-label">SNR</div></div>' +
        '</div>';
    });

    if (data.mpc) {
      document.getElementById('mpcSection').style.display='block';
      document.getElementById('mpcBox').textContent = data.mpc;
    }

    var grid = document.getElementById('imgGrid');
    grid.innerHTML = '';
    if (data.plot_detection) {
      document.getElementById('imgSection').style.display='block';
      grid.innerHTML += '<div class="img-card"><img src="data:image/png;base64,'+data.plot_detection+'"><div class="img-label">deteccoes</div></div>';
    }
    if (data.plot_blink) {
      grid.innerHTML += '<div class="img-card"><img src="data:image/png;base64,'+data.plot_blink+'"><div class="img-label">blink comparator</div></div>';
    }

    log('[OK] Pipeline concluido -- '+data.candidates.length+' candidato(s)', 'lok');
  }

  function copyMpc() {
    var txt = document.getElementById('mpcBox').textContent;
    navigator.clipboard.writeText(txt).then(function(){
      var b=document.querySelector('.copy-btn');
      b.textContent='copiado!';
      setTimeout(function(){b.textContent='copiar';},1500);
    });
  }
</script>
</body>
</html>'''


# ---------------------------------------------
#  ROTAS FLASK
# ---------------------------------------------

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/run', methods=['POST'])
def run():
    if not PIPELINE_OK:
        return jsonify({'error': f'asteroid_detector.py nao encontrado: {PIPELINE_ERROR}'}), 500

    log_lines = []

    class LogCapture(logging.Handler):
        def emit(self, record):
            log_lines.append(self.format(record))

    handler = LogCapture()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S'))
    logging.getLogger().addHandler(handler)

    tmp_dir = tempfile.mkdtemp()
    fits_paths = []

    try:
        # Salva os FITS enviados
        for f in request.files.getlist('fits_files'):
            path = os.path.join(tmp_dir, f.filename)
            f.save(path)
            fits_paths.append(path)

        if len(fits_paths) < 1:
            return jsonify({'error': 'Nenhum arquivo FITS recebido'}), 400

        # Parametros
        config = {
            'fwhm':           float(request.form.get('fwhm', 3.0)),
            'threshold_sigma': float(request.form.get('threshold', 8.0)),
            'match_radius':    5.0,
            'min_motion_px':  float(request.form.get('min_motion', 2.0)),
            'max_motion_px':  float(request.form.get('max_motion', 20.0)),
            'min_snr':        float(request.form.get('min_snr', 8.0)),
        }
        obs_code = request.form.get('obs_code', '500')
        do_plot  = request.form.get('plot',  '1') == '1'
        do_mpc   = request.form.get('mpc',   '1') == '1'
        do_blink = request.form.get('blink', '1') == '1'

        # Roda o pipeline
        pipeline = AsteroidDetectionPipeline(config)
        results  = pipeline.run(fits_paths)
        candidates = results['candidates']

        # Formato MPC
        mpc_text = None
        if do_mpc and candidates:
            astrometry = Astrometry()
            mpc_cands = [
                {'ra': c['ra_deg'], 'dec': c['dec_deg'],
                 'magnitude': round(99.9 - min(c['snr'] / 2, 10), 1)}
                for c in candidates if c['ra_deg'] is not None
            ]
            obs_time = candidates[0].get('obs_time') if candidates else None
            if mpc_cands:
                mpc_text = astrometry.format_mpc(mpc_cands, obs_code, obs_time)

        # Graficos em base64
        plot_b64 = None
        blink_b64 = None

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        from astropy.visualization import ZScaleInterval, ImageNormalize

        if do_plot and candidates:
            loader = FITSLoader(fits_paths[0])
            data_sub, _ = pipeline.preprocessor.subtract_background(loader.data)
            while data_sub.ndim > 2:
                data_sub = data_sub[0]

            plot_cands = [{**c, 'x': c['x_px'], 'y': c['y_px']} for c in candidates]

            fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0a0f')
            ax.set_facecolor('#0a0a0f')
            norm = ImageNormalize(data_sub, interval=ZScaleInterval())
            ax.imshow(data_sub, cmap='gray', norm=norm, origin='lower')
            for i, cand in enumerate(plot_cands):
                x, y = cand['x'], cand['y']
                circle = patches.Circle((x, y), radius=12, linewidth=1.5,
                    edgecolor='#00ff88', facecolor='none', linestyle='--')
                ax.add_patch(circle)
                ax.text(x+14, y, f'#{i+1}', color='#00ff88', fontsize=8,
                    va='center', fontfamily='monospace')
            ax.set_title(f'{len(plot_cands)} candidato(s) detectado(s)', color='white', fontsize=13)
            ax.tick_params(colors='gray')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0a0a0f')
            plt.close(fig)
            buf.seek(0)
            plot_b64 = base64.b64encode(buf.read()).decode()

        if do_blink and len(fits_paths) > 1 and candidates:
            from astropy.io import fits as afits
            from astropy.visualization import ZScaleInterval, ImageNormalize
            frames_data = []
            for fp in fits_paths[:4]:
                with afits.open(fp) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None:
                            d = hdu.data.astype(np.float64)
                            while d.ndim > 2: d = d[0]
                            frames_data.append(d)
                            break

            plot_cands = [{**c, 'x': c['x_px'], 'y': c['y_px']} for c in candidates]
            n = min(len(frames_data), 4)
            fig2, axes = plt.subplots(1, n, figsize=(5*n, 5), facecolor='#0a0a0f')
            if n == 1: axes = [axes]
            for i, (ax, data) in enumerate(zip(axes, frames_data[:n])):
                ax.set_facecolor('#0a0a0f')
                norm = ImageNormalize(data, interval=ZScaleInterval())
                ax.imshow(data, cmap='gray', norm=norm, origin='lower')
                ax.set_title(f'Frame {i+1}', color='white', fontsize=10)
                ax.tick_params(colors='gray', labelsize=7)
                for cand in plot_cands:
                    circle = patches.Circle((cand['x'], cand['y']), radius=10,
                        linewidth=1.5, edgecolor='#ff4444', facecolor='none')
                    ax.add_patch(circle)
            plt.suptitle('Blink Comparator', color='white', fontsize=12)
            plt.tight_layout()
            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png', dpi=120, bbox_inches='tight', facecolor='#0a0a0f')
            plt.close(fig2)
            buf2.seek(0)
            blink_b64 = base64.b64encode(buf2.read()).decode()

        return jsonify({
            'candidates':      candidates,
            'mpc':             mpc_text,
            'plot_detection':  plot_b64,
            'plot_blink':      blink_b64,
            'log':             log_lines,
            'n_frames':        results['n_frames'],
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'log': log_lines}), 500

    finally:
        logging.getLogger().removeHandler(handler)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------
#  MAIN
# ---------------------------------------------

if __name__ == '__main__':
    print('\n' + '='*50)
    print('  AstroDetect -- Servidor Flask')
    print('  Acesse: http://localhost:5000')
    print('='*50 + '\n')
    port = int(os.environ.get('PORT', 5000))
app.run(debug=False, host='0.0.0.0', port=port)
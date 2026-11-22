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

HTML = r'''<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AstroDetect | Artemis Aerospace</title>
<link href="https://fonts.googleapis.com/css2?family=Oxanium:wght@400;600;800&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#070f24;--panel:#0d1a3a;--panel2:#112247;--line:#2a3b63;--silver:#c8d0de;--dim:#8f9bb3;--gold:#e3b341;--gold2:#f3d27a;--ink:#eef2fa;--bad:#e5604f}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Oxanium',sans-serif;color:var(--ink);min-height:100vh;padding:28px 20px 48px;background:radial-gradient(1200px 500px at 50% -10%,#16295a 0%,transparent 70%),radial-gradient(2px 2px at 12% 20%,#fff8 50%,transparent 51%),radial-gradient(1.5px 1.5px at 78% 12%,#fff7 50%,transparent 51%),radial-gradient(1.5px 1.5px at 90% 60%,#fff6 50%,transparent 51%),radial-gradient(2px 2px at 30% 80%,#fff6 50%,transparent 51%),var(--bg)}
.wrap{max-width:1180px;margin:0 auto}
header{display:flex;align-items:center;gap:22px;padding-bottom:22px;margin-bottom:24px;border-bottom:1px solid var(--line);position:relative}
header::after{content:"";position:absolute;left:0;bottom:-1px;width:160px;height:2px;background:linear-gradient(90deg,var(--gold),transparent)}
h1{font-size:clamp(34px,5vw,54px);font-weight:800;letter-spacing:.5px;line-height:1;background:linear-gradient(180deg,#fff 10%,#aab4c8 55%,#e9edf5 90%);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{margin-top:8px;color:var(--dim);font-size:15px}
.sub b{color:var(--gold2);font-weight:600}
.state{margin-left:auto;font-size:13px;color:var(--gold2);border:1px solid var(--gold);border-radius:99px;padding:6px 14px;white-space:nowrap}
.grid{display:grid;grid-template-columns:400px 1fr;gap:24px;align-items:start}
.panel{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-top:2px solid var(--gold);border-radius:6px;padding:20px}
.panel+.panel{margin-top:18px}
h2{font-size:17px;font-weight:600;color:var(--silver);margin-bottom:14px}
.dz{position:relative;border:1.5px dashed #4a5f91;border-radius:6px;padding:26px 16px;text-align:center;background:#0a1531;transition:.2s}
.dz:hover,.dz.over{border-color:var(--gold);background:#0f1d40}
.dz input{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer}
.dz b{display:block;font-size:17px;margin-bottom:4px}
.dz span{font-size:13px;color:var(--dim)}
.fl{margin-top:10px;display:flex;flex-direction:column;gap:6px}
.f{display:flex;gap:10px;align-items:center;background:#0a1531;border:1px solid var(--line);border-radius:4px;padding:8px 12px;font:13px 'IBM Plex Mono',monospace}
.f i{color:var(--gold);font-style:normal}.f em{flex:1;font-style:normal;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.f small{color:var(--dim)}
.f button{background:none;border:0;color:var(--dim);cursor:pointer;font-size:15px}.f button:hover{color:var(--bad)}
.ps{display:grid;grid-template-columns:1fr 1fr;gap:12px}
label.p{display:flex;flex-direction:column;gap:5px;font-size:13px;color:var(--dim)}
input[type=number],input[type=text]{width:100%;background:#0a1531;border:1px solid var(--line);border-radius:4px;padding:8px 10px;font:14px 'IBM Plex Mono',monospace;color:var(--ink)}
input:focus-visible,button:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
.ck{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:16px;font-size:14px;color:var(--silver)}
.ck label{display:flex;gap:7px;align-items:center;cursor:pointer}.ck input{accent-color:var(--gold)}
.run{width:100%;margin-top:18px;padding:14px;border:0;border-radius:4px;font:700 17px 'Oxanium',sans-serif;color:#1a1204;cursor:pointer;background:linear-gradient(180deg,var(--gold2),var(--gold));box-shadow:0 6px 18px #e3b34130;transition:.15s}
.run:hover{filter:brightness(1.08)}.run:active{transform:translateY(1px)}.run:disabled{filter:grayscale(.8) brightness(.6);cursor:wait}
.bar{height:3px;background:#0a1531;margin-top:12px;border-radius:2px;overflow:hidden;display:none}.bar i{display:block;height:100%;width:0;background:var(--gold);transition:width .4s}
.log{display:none;margin-top:12px;max-height:150px;overflow:auto;background:#050b1c;border:1px solid var(--line);border-radius:4px;padding:10px;font:11.5px/1.7 'IBM Plex Mono',monospace;color:#7f8fb0}
.log .w{color:var(--gold2)}.log .k{color:#6fd6a0}
.err{display:none;margin-top:12px;padding:12px;border:1px solid var(--bad);border-radius:4px;color:#ffb1a6;font:13px 'IBM Plex Mono',monospace;white-space:pre-wrap}
.empty{padding:46px 20px;text-align:center;color:var(--dim);line-height:1.6}
.empty b{display:block;color:var(--silver);font-size:18px;margin-bottom:6px}
.head{display:flex;align-items:center;gap:10px;margin-bottom:14px}.head h2{margin:0}
.n{background:var(--gold);color:#1a1204;font-weight:800;border-radius:99px;padding:1px 11px;font-size:15px}
.c{display:grid;grid-template-columns:40px 1fr auto;gap:14px;align-items:center;background:#0a1531;border:1px solid var(--line);border-left:3px solid var(--gold);border-radius:4px;padding:12px 14px;margin-bottom:8px}
.c .i{width:34px;height:34px;border-radius:50%;background:var(--gold);color:#1a1204;font-weight:800;display:grid;place-items:center}
.c .r{font:14px 'IBM Plex Mono',monospace;color:var(--ink)}.c .m{font-size:13px;color:var(--dim);margin-top:3px}
.c .s{text-align:right;font:600 22px 'Oxanium';color:var(--gold2)}.c .s small{display:block;font-size:12px;color:var(--dim);font-weight:400}
.mpc{background:#050b1c;border:1px solid var(--line);border-radius:4px;padding:14px;font:13px/1.9 'IBM Plex Mono',monospace;color:var(--gold2);white-space:pre;overflow-x:auto}
.cp{margin-left:auto;background:none;border:1px solid var(--line);color:var(--silver);border-radius:4px;padding:5px 12px;font:13px 'Oxanium';cursor:pointer}.cp:hover{border-color:var(--gold);color:var(--gold2)}
.im{margin-top:14px}.im img{width:100%;display:block;border:1px solid var(--line);border-radius:4px;background:#050b1c}.im p{font-size:13px;color:var(--dim);margin-bottom:6px}
@media(max-width:900px){.grid{grid-template-columns:1fr}header{flex-wrap:wrap}.state{margin-left:0}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>AstroDetect</h1>
    <div class="sub"><b>Artemis Aerospace</b> &middot; Universidade Estadual do Cear&aacute;</div>
  </div>
  <div class="state" id="state">Pronto para analisar</div>
</header>

<div class="grid">
  <div>
    <section class="panel">
      <h2>Frames FITS</h2>
      <div class="dz" id="dz">
        <input type="file" id="fi" multiple accept=".fits,.fit" onchange="handleFiles(this.files);this.value=''">
        <b>Solte os arquivos aqui</b>
        <span>ou clique para escolher. Use 3 frames ou mais do mesmo campo.</span>
      </div>
      <div class="fl" id="fl"></div>
    </section>

    <section class="panel">
      <h2>Par&acirc;metros de detec&ccedil;&atilde;o</h2>
      <div class="ps">
        <label class="p">Limiar (sigma)<input type="number" id="threshold" value="8.0" step="0.5" min="1"></label>
        <label class="p">FWHM (px)<input type="number" id="fwhm" value="3.0" step="0.5" min="1"></label>
        <label class="p">Movimento m&iacute;nimo (px)<input type="number" id="minMotion" value="2.0" step="0.5" min="0"></label>
        <label class="p">Movimento m&aacute;ximo (px)<input type="number" id="maxMotion" value="20.0" step="1" min="1"></label>
        <label class="p">SNR m&iacute;nimo<input type="number" id="minSnr" value="8.0" step="0.5" min="1"></label>
        <label class="p">C&oacute;digo MPC<input type="text" id="obsCode" value="500" maxlength="3"></label>
      </div>
      <div class="ck">
        <label><input type="checkbox" id="chkPlot" checked>Gr&aacute;fico de detec&ccedil;&otilde;es</label>
        <label><input type="checkbox" id="chkMpc" checked>Formato MPC</label>
        <label><input type="checkbox" id="chkBlink" checked>Blink comparator</label>
      </div>
      <button class="run" id="run" onclick="runPipeline()">Analisar frames</button>
      <div class="bar" id="bar"><i id="pb"></i></div>
      <div class="log" id="log"></div>
      <div class="err" id="err"></div>
    </section>
  </div>

  <section class="panel" id="out">
    <div class="empty" id="empty"><b>Nenhuma an&aacute;lise ainda</b>Envie os frames e clique em Analisar. Os candidatos, o formato MPC e os gr&aacute;ficos aparecem aqui.</div>
    <div id="res" style="display:none">
      <div class="head"><h2>Candidatos</h2><span class="n" id="cnt">0</span></div>
      <div id="cl"></div>
      <div id="ms" style="display:none;margin-top:20px">
        <div class="head"><h2>Formato MPC</h2><button class="cp" onclick="copyMpc(this)">Copiar</button></div>
        <div class="mpc" id="mpc"></div>
      </div>
      <div id="is"></div>
    </div>
  </section>
</div>
</div>

<script>
var files=[];
function $(i){return document.getElementById(i)}
function handleFiles(f){for(var i=0;i<f.length;i++)files.push(f[i]);renderFiles()}
function removeFile(i){files.splice(i,1);renderFiles()}
function renderFiles(){
  $('fl').innerHTML=files.map(function(f,i){
    return '<div class="f"><i>&#9670;</i><em>'+f.name+'</em><small>'+(f.size/1048576).toFixed(1)+' MB</small><button onclick="removeFile('+i+')" aria-label="Remover">&times;</button></div>';
  }).join('');
}
var dz=$('dz');
dz.addEventListener('dragover',function(e){e.preventDefault();dz.classList.add('over')});
dz.addEventListener('dragleave',function(){dz.classList.remove('over')});
dz.addEventListener('drop',function(e){e.preventDefault();dz.classList.remove('over');handleFiles(e.dataTransfer.files)});

function logLine(t){
  var b=$('log');b.style.display='block';
  var c=t.indexOf('WARNING')>-1?'w':t.indexOf('concluido')>-1?'k':'';
  var d=document.createElement('div');d.className=c;d.textContent=t;b.appendChild(d);b.scrollTop=b.scrollHeight;
}
function prog(p){$('bar').style.display='block';$('pb').style.width=p+'%'}
function fail(msg){var e=$('err');e.textContent=msg;e.style.display='block'}

async function runPipeline(){
  if(files.length<3){fail('Selecione pelo menos 3 arquivos FITS do mesmo campo.');return}
  var btn=$('run');btn.disabled=true;btn.textContent='Analisando...';$('state').textContent='Processando';
  $('log').innerHTML='';$('log').style.display='none';$('err').style.display='none';prog(10);
  var fd=new FormData();
  files.forEach(function(f){fd.append('fits_files',f)});
  fd.append('threshold',$('threshold').value);fd.append('fwhm',$('fwhm').value);
  fd.append('min_motion',$('minMotion').value);fd.append('max_motion',$('maxMotion').value);
  fd.append('min_snr',$('minSnr').value);fd.append('obs_code',$('obsCode').value);
  fd.append('plot',$('chkPlot').checked?'1':'0');fd.append('mpc',$('chkMpc').checked?'1':'0');fd.append('blink',$('chkBlink').checked?'1':'0');
  logLine('Enviando '+files.length+' frames...');prog(25);
  try{
    var r=await fetch('/run',{method:'POST',body:fd});prog(85);
    var d=await r.json();prog(100);
    if(d.error){fail('Erro na an\u00e1lise: '+d.error)}
    else{(d.log||[]).forEach(logLine);show(d)}
  }catch(e){fail('Sem conex\u00e3o com o servidor: '+e.message)}
  btn.disabled=false;btn.textContent='Analisar frames';$('state').textContent='Pronto para analisar';
}

function show(d){
  $('empty').style.display='none';$('res').style.display='block';
  $('cnt').textContent=d.candidates.length;
  $('cl').innerHTML=d.candidates.length?d.candidates.map(function(c,i){
    var ra=c.ra_deg!=null?c.ra_deg.toFixed(6):'N/A',dec=c.dec_deg!=null?c.dec_deg.toFixed(6):'N/A';
    return '<div class="c"><div class="i">'+(i+1)+'</div><div><div class="r">RA '+ra+'&deg; &nbsp; Dec '+dec+'&deg;</div><div class="m">moveu '+c.motion_px+' px na dire\u00e7\u00e3o '+c.motion_angle_deg+'&deg;</div></div><div class="s">'+c.snr.toFixed(1)+'<small>SNR</small></div></div>';
  }).join(''):'<div class="empty"><b>Nenhum candidato</b>Tente baixar o limiar, o SNR m\u00ednimo ou o movimento m\u00ednimo.</div>';
  $('ms').style.display=d.mpc?'block':'none';
  if(d.mpc)$('mpc').textContent=d.mpc;
  var h='';
  if(d.plot_detection)h+='<div class="im"><p>Detec\u00e7\u00f5es no primeiro frame</p><img src="data:image/png;base64,'+d.plot_detection+'" alt="Detec\u00e7\u00f5es"></div>';
  if(d.plot_blink)h+='<div class="im"><p>Blink comparator</p><img src="data:image/png;base64,'+d.plot_blink+'" alt="Blink comparator"></div>';
  $('is').innerHTML=h;
}
function copyMpc(b){
  navigator.clipboard.writeText($('mpc').textContent).then(function(){b.textContent='Copiado';setTimeout(function(){b.textContent='Copiar'},1500)});
}
</script>
</body>
</html>
'''


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

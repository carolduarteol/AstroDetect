import requests
import time

candidatos = [
    {"ra": 55.514208, "dec": 9.744472},
    {"ra": 55.513167, "dec": 9.745583},
    {"ra": 55.512250, "dec": 9.749472},
    {"ra": 55.512417, "dec": 9.749611},
]

url_base = (
    "https://ssd-api.jpl.nasa.gov/sb_ident.api"
    "?obs-time=2026-03-26"
    "&ra-deg={ra}&dec-deg={dec}"
    "&radius=0.5&mag-required=false"
    "&two-pass=true&suppress-first-pass=true"
)

print("Verificando candidatos no catálogo JPL...\n")

for i, c in enumerate(candidatos):
    url = url_base.format(ra=c["ra"], dec=c["dec"])
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        objetos = data.get("data", [])
        print(f"Candidato #{i+1} — RA={c['ra']} Dec={c['dec']}")
        if objetos:
            for obj in objetos:
                print(f"  ⚠️  CONHECIDO: {obj}")
        else:
            print(f"  ✅ Não encontrado no catálogo!")
    except Exception as e:
        print(f"  ❌ Erro: {e}")
    time.sleep(2)
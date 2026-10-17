"""
Web Scraper e Pipeline de Ingestão e Classificação por IA (AcademicIA).
Extrai editais e documentos de processos seletivos dos portais oficiais
das 4 instituições públicas de ensino superior do Piauí:
  - UFPI (SIGAA)
  - UESPI (SIGAA)
  - IFPI (Portal Plone)
  - UFDPAR (SIGAA)

Organização hierárquica estrita dos dados:
  data/scraped/{faculdade}/{curso}/{nome_arquivo}.pdf
  data/scraped/{faculdade}/{curso}/{nome_arquivo}.meta.json
  data/scraped/catalogo_documentos.json (catálogo centralizado)

Cada documento baixado passa automaticamente pelo modelo de Deep Learning
(models/best.pt) para classificação em tempo de ingestão (6 classes).
"""
import sys
import io
import re
import json
import time
import argparse
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import torch
import urllib3

# Desativa avisos de SSL em portais governamentais com certificados internos
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuração de encoding UTF-8 segura para o Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CURSOS_JSON = BASE_DIR / "data" / "cursos_piaui.json"
MODELS_DIR = BASE_DIR / "models"
MODEL_CHECKPOINT = MODELS_DIR / "best.pt"
VOCAB_PATH = MODELS_DIR / "vocab.json"
DEFAULT_SCRAPED_DIR = BASE_DIR / "data" / "scraped"

ID_TO_LABEL = {
    0: "edital",
    1: "homologacao",
    2: "recurso",
    3: "resultado",
    4: "aviso",
    5: "retificacao"
}

PALAVRAS_CHAVE_EDITAIS = [
    "edital", "selecao", "seleção", "resultado", "retificacao", "retificação",
    "homologacao", "homologação", "recurso", "convocacao", "convocação",
    "chamada", "inscric", "inscrição", "gabarito", "admissao", "admissão"
]


def sanitizar_nome(nome: str, max_chars: int = 65) -> str:
    """Normaliza e sanitiza nomes para caminhos válidos de pastas e arquivos no SO."""
    if not nome:
        return "sem_nome"
    nfkd = unicodedata.normalize("NFKD", nome)
    sem_acento = "".join([c for c in nfkd if not unicodedata.combining(c)])
    limpo = re.sub(r"[^\w\s-]", "_", sem_acento)
    limpo = re.sub(r"[\s_]+", "_", limpo).strip("_")
    return limpo[:max_chars]


def limpar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = re.sub(r"[\r\n\t]+", " ", texto)
    texto = re.sub(r"\s{2,}", " ", texto)
    return texto.strip()


class IngestClassifier:
    """Carrega o modelo PyTorch treinado e realiza inferência em tempo de ingestão."""
    def __init__(self, model_path: Path = MODEL_CHECKPOINT, vocab_path: Path = VOCAB_PATH):
        self.device = torch.device("cpu")
        self.available = False
        
        if not model_path.exists() or not vocab_path.exists():
            print(f"[!] Aviso: Modelo ou vocabulário não encontrados. Modo sem inferência.", flush=True)
            return

        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vdata = json.load(f)
            self.w2i = vdata["w2i"]
            self.pad_id = self.w2i.get("<pad>", 0)
            self.unk_id = self.w2i.get("<unk>", 1)
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            sys.path.insert(0, str(BASE_DIR))
            from src.models.network import TextClassifier
            
            emb_dim = checkpoint.get("embedding_dim", 128)
            hidden_dim = checkpoint.get("hidden_dim", 64)
            num_classes = checkpoint.get("num_classes", 6)
            
            self.model = TextClassifier(
                vocab_size=len(self.w2i),
                embedding_dim=emb_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout_prob=0.0
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.available = True
            print(f"[OK] Modelo PyTorch carregado para inferência em tempo real ({model_path.name})", flush=True)
        except Exception as e:
            print(f"[!] Erro ao inicializar classificador: {e}", flush=True)
            self.available = False

    def encode(self, texto: str, max_len: int = 128) -> list[int]:
        palavras = re.findall(r"\b\w+\b", texto.lower())
        ids = [self.w2i.get(p, self.unk_id) for p in palavras[:max_len]]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def extrair_texto_pdf(self, pdf_path: Path, max_chars: int = 1000) -> tuple[str, int]:
        try:
            reader = PdfReader(str(pdf_path))
            num_paginas = len(reader.pages)
            if num_paginas == 0:
                return "", 0
            texto = reader.pages[0].extract_text() or ""
            if len(limpar_texto(texto)) < 80 and num_paginas > 1:
                texto += " " + (reader.pages[1].extract_text() or "")
            return limpar_texto(texto)[:max_chars], num_paginas
        except Exception as e:
            print(f"    [!] Erro ao extrair texto do PDF {pdf_path.name}: {e}", flush=True)
            return "", 0

    def classificar(self, pdf_path: Path) -> dict:
        if not self.available:
            return {"classe_predita": "desconhecido", "confianca_percentual": 0.0, "erro": "modelo_indisponivel"}
            
        texto, total_paginas = self.extrair_texto_pdf(pdf_path)
        if not texto:
            return {
                "classe_predita": "indeterminado",
                "confianca_percentual": 0.0,
                "paginas": total_paginas,
                "motivo": "pdf_sem_texto_extraivel"
            }
            
        token_ids = self.encode(texto, max_len=128)
        tensor_in = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_label = ID_TO_LABEL.get(pred_idx, "desconhecido")
            conf = probs[pred_idx]
            
        distribuicao = {ID_TO_LABEL[i]: round(probs[i] * 100, 2) for i in range(len(probs))}
        
        return {
            "classe_predita": pred_label,
            "confianca_percentual": round(conf * 100, 2),
            "paginas": total_paginas,
            "distribuicao_probabilidades": distribuicao,
            "trecho_inicial": texto[:250] + ("..." if len(texto) > 250 else "")
        }


class WebScraperAcademicIA:
    """Orquestrador do Web Scraping para as 4 Universidades do Piauí."""
    def __init__(self, output_dir: Path = DEFAULT_SCRAPED_DIR, delay: float = 0.5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8"
        })
        self.session.verify = False
        
        self.classifier = IngestClassifier()
        self.catalogo_path = self.output_dir / "catalogo_documentos.json"
        self.catalogo = self._carregar_catalogo()

    def _carregar_catalogo(self) -> list[dict]:
        if self.catalogo_path.exists():
            try:
                with open(self.catalogo_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _salvar_catalogo(self):
        with open(self.catalogo_path, "w", encoding="utf-8") as f:
            json.dump(self.catalogo, f, ensure_ascii=False, indent=2)

    def extrair_candidatos_sigaa(self, portal_url: str) -> list[dict]:
        """
        Navega de forma eficiente no SIGAA:
        1. Varre documentos.jsf (onde editais e resoluções ficam agrupados)
        2. Varre portal.jsf
        3. Varre noticias.jsf buscando apenas as notícias com palavras-chave relevantes
        """
        candidatos = []
        parsed = urlparse(portal_url)
        prog_id = parse_qs(parsed.query).get("id", [""])[0]

        # 1. documentos.jsf é a fonte mais rica de anexos no SIGAA
        if prog_id:
            doc_url = urljoin(portal_url, f"documentos.jsf?lc=pt_BR&id={prog_id}")
            try:
                resp = self.session.get(doc_url, timeout=7)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for a in soup.find_all("a", href=True):
                        raw_href = a.get("href", "")
                        clean_href = re.sub(r"[\r\n\t\s]+", "", raw_href)
                        titulo = a.get_text(strip=True) or a.get("title", "") or "documento"
                        if "verProducao" in clean_href or clean_href.lower().endswith(".pdf"):
                            candidatos.append({
                                "url": urljoin(doc_url, clean_href),
                                "titulo": titulo,
                                "origem_pagina": doc_url
                            })
            except Exception:
                pass

        # 2. portal.jsf principal
        try:
            resp = self.session.get(portal_url, timeout=7)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    raw_href = a.get("href", "")
                    clean_href = re.sub(r"[\r\n\t\s]+", "", raw_href)
                    titulo = a.get_text(strip=True) or "documento"
                    if "verProducao" in clean_href or clean_href.lower().endswith(".pdf"):
                        candidatos.append({
                            "url": urljoin(portal_url, clean_href),
                            "titulo": titulo,
                            "origem_pagina": portal_url
                        })
        except Exception:
            pass

        # 3. noticias.jsf (filtragem focada por notícias relevantes de seleção)
        if prog_id and len(candidatos) < 10:
            noticias_url = urljoin(portal_url, f"noticias.jsf?lc=pt_BR&id={prog_id}")
            try:
                resp = self.session.get(noticias_url, timeout=7)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    links_noticia = []
                    for a in soup.find_all("a", href=True):
                        raw_href = a.get("href", "")
                        clean_href = re.sub(r"[\r\n\t\s]+", "", raw_href)
                        text = a.get_text(strip=True)
                        if "verProducao" in clean_href or clean_href.lower().endswith(".pdf"):
                            candidatos.append({
                                "url": urljoin(noticias_url, clean_href),
                                "titulo": text or "Documento Anexo",
                                "origem_pagina": noticias_url
                            })
                        elif "noticias_desc.jsf" in clean_href:
                            if any(k in text.lower() for k in PALAVRAS_CHAVE_EDITAIS):
                                links_noticia.append((urljoin(noticias_url, clean_href), text))

                    # Inspeciona no máximo 3 notícias relevantes com editais
                    for n_url, n_title in links_noticia[:3]:
                        try:
                            time.sleep(0.3)
                            n_resp = self.session.get(n_url, timeout=6)
                            if n_resp.status_code == 200:
                                n_soup = BeautifulSoup(n_resp.text, "html.parser")
                                for na in n_soup.find_all("a", href=True):
                                    n_raw = na.get("href", "")
                                    n_clean = re.sub(r"[\r\n\t\s]+", "", n_raw)
                                    if "verProducao" in n_clean or n_clean.lower().endswith(".pdf"):
                                        candidatos.append({
                                            "url": urljoin(n_url, n_clean),
                                            "titulo": na.get_text(strip=True) or n_title,
                                            "origem_pagina": n_url
                                        })
                        except Exception:
                            pass
            except Exception:
                pass

        # Ordena candidatos priorizando termos de editais e processos seletivos
        def pontuacao_relevancia(cand):
            tit = cand["titulo"].lower()
            return sum(2 for k in PALAVRAS_CHAVE_EDITAIS if k in tit)

        candidatos.sort(key=pontuacao_relevancia, reverse=True)

        vistos = set()
        unicos = []
        for c in candidatos:
            if c["url"] not in vistos:
                vistos.add(c["url"])
                unicos.append(c)
        return unicos

    def extrair_candidatos_ifpi(self, portal_url: str) -> list[dict]:
        """Navega no portal Plone do IFPI e captura documentos diretos e páginas de seleção."""
        candidatos = []
        try:
            resp = self.session.get(portal_url, timeout=8)
            if resp.status_code != 200:
                return []
            
            soup = BeautifulSoup(resp.text, "html.parser")
            sub_links = []
            
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                text = a.get_text(strip=True)
                full_url = urljoin(portal_url, href)
                
                if ".pdf" in full_url.lower():
                    clean_pdf_url = full_url
                    if clean_pdf_url.endswith("/view"):
                        clean_pdf_url = clean_pdf_url[:-5]
                    candidatos.append({
                        "url": clean_pdf_url,
                        "titulo": text or "Documento IFPI",
                        "origem_pagina": portal_url
                    })
                elif any(k in href.lower() for k in ["processo-seletivo", "selecao", "editais", "ingresso", "o-mestrado"]):
                    if full_url.startswith("https://www.ifpi.edu.br"):
                        sub_links.append((full_url, text))
                        
            for sub_url, sub_title in sub_links[:2]:
                try:
                    time.sleep(0.4)
                    s_resp = self.session.get(sub_url, timeout=7)
                    if s_resp.status_code == 200:
                        s_soup = BeautifulSoup(s_resp.text, "html.parser")
                        for sa in s_soup.find_all("a", href=True):
                            shref = sa.get("href")
                            if ".pdf" in shref.lower():
                                s_pdf = urljoin(sub_url, shref)
                                if s_pdf.endswith("/view"):
                                    s_pdf = s_pdf[:-5]
                                candidatos.append({
                                    "url": s_pdf,
                                    "titulo": sa.get_text(strip=True) or sub_title,
                                    "origem_pagina": sub_url
                                })
                except Exception:
                    pass
        except Exception as e:
            print(f"    [!] Erro ao varrer IFPI ({portal_url}): {e}", flush=True)

        vistos = set()
        unicos = []
        for c in candidatos:
            if c["url"] not in vistos:
                vistos.add(c["url"])
                unicos.append(c)
        return unicos

    def baixar_documento(self, url: str, destino: Path) -> bool:
        """Faz o download com validação da assinatura mágica do arquivo PDF."""
        try:
            resp = self.session.get(url, stream=True, timeout=15)
            if resp.status_code != 200:
                return False
            
            chunk = resp.raw.read(1024)
            if not chunk or b"%PDF" not in chunk[:128]:
                return False
                
            with open(destino, "wb") as f:
                f.write(chunk)
                for block in resp.iter_content(chunk_size=32768):
                    if block:
                        f.write(block)
            return True
        except Exception:
            if destino.exists():
                try:
                    destino.unlink()
                except Exception:
                    pass
            return False

    def coletar_curso(self, faculdade: str, curso_info: dict, max_docs: int = 2) -> int:
        """Coleta documentos de um curso e salva na hierarquia: data/scraped/{faculdade}/{curso}/"""
        nome_curso = curso_info.get("curso", "curso_desconhecido")
        portal_link = curso_info.get("link", "")
        
        curso_folder_name = sanitizar_nome(nome_curso)
        curso_dir = self.output_dir / faculdade / curso_folder_name
        curso_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📂 [{faculdade}] {nome_curso[:65]}", flush=True)
        print(f"   🔗 Link: {portal_link}", flush=True)
        print(f"   📁 Pasta: {curso_dir.relative_to(BASE_DIR)}", flush=True)

        if faculdade in ["UFPI", "UESPI", "UFDPAR"]:
            candidatos = self.extrair_candidatos_sigaa(portal_link)
        elif faculdade == "IFPI":
            candidatos = self.extrair_candidatos_ifpi(portal_link)
        else:
            candidatos = []

        print(f"   🔎 Documentos candidatos identificados: {len(candidatos)}", flush=True)

        docs_baixados = 0
        for idx, item in enumerate(candidatos):
            if docs_baixados >= max_docs:
                break
                
            doc_url = item["url"]
            titulo = item["titulo"]
            
            nome_base = sanitizar_nome(titulo) or f"doc_{idx+1}"
            pdf_filename = f"{nome_base}.pdf"
            meta_filename = f"{nome_base}.meta.json"
            
            pdf_path = curso_dir / pdf_filename
            meta_path = curso_dir / meta_filename
            rel_path = str(pdf_path.relative_to(BASE_DIR)).replace("\\", "/")
            
            if pdf_path.exists() and pdf_path.stat().st_size > 500:
                print(f"   ⚡ [{docs_baixados+1}/{max_docs}] Já existente: {pdf_filename}", flush=True)
                docs_baixados += 1
                continue
                
            print(f"   ⬇️  [{docs_baixados+1}/{max_docs}] Baixando: {titulo[:45]}...", flush=True)
            sucesso = self.baixar_documento(doc_url, pdf_path)
            
            if not sucesso:
                print(f"      ⏭️  Arquivo não é PDF ou indisponível. Tentando próximo candidato...", flush=True)
                continue

            tamanho_kb = pdf_path.stat().st_size / 1024
            print(f"      ✅ Salvo com sucesso: {tamanho_kb:.1f} KB", flush=True)

            # Classificação IA em tempo real com o modelo treinado
            classificacao = self.classifier.classificar(pdf_path)
            classe = classificacao.get("classe_predita", "desconhecido")
            conf = classificacao.get("confianca_percentual", 0.0)
            print(f"      🧠 Predição IA: [{classe.upper()}] | Confiança: {conf}%", flush=True)

            meta_data = {
                "faculdade": faculdade,
                "curso": nome_curso,
                "pasta_curso": curso_folder_name,
                "nivel": curso_info.get("nivel", ""),
                "cidade": curso_info.get("cidade", ""),
                "arquivo_pdf": pdf_filename,
                "caminho_relativo": rel_path,
                "tamanho_bytes": pdf_path.stat().st_size,
                "url_documento": doc_url,
                "url_portal": portal_link,
                "data_coleta": datetime.now().isoformat(),
                "classificacao_ia": classificacao
            }
            
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)

            # Atualiza o catálogo evitando duplicatas
            self.catalogo = [c for c in self.catalogo if c.get("caminho_relativo") != rel_path]
            self.catalogo.append(meta_data)
            self._salvar_catalogo()
            docs_baixados += 1
            
            time.sleep(self.delay)

        return docs_baixados

    def executar(self, faculdades: list[str] = None, limit_cursos_por_faculdade: int = 2, max_docs_por_curso: int = 2):
        if not CURSOS_JSON.exists():
            print(f"[!] Erro: Arquivo de cursos não encontrado em {CURSOS_JSON}", flush=True)
            return

        with open(CURSOS_JSON, "r", encoding="utf-8") as f:
            banco_cursos = json.load(f)

        faculdades_alvo = faculdades or list(banco_cursos.keys())
        total_baixados_geral = 0
        inicio = time.time()

        print("=" * 75, flush=True)
        print("  INICIANDO PIPELINE DE WEB SCRAPING E CLASSIFICAÇÃO - AcademicIA", flush=True)
        print(f"  Faculdades alvo: {', '.join(faculdades_alvo)}", flush=True)
        print(f"  Limite por instituição: {'Todos os cursos' if limit_cursos_por_faculdade == 0 else f'{limit_cursos_por_faculdade} cursos'}", flush=True)
        print(f"  Max documentos por curso: {max_docs_por_curso}", flush=True)
        print(f"  Estrutura de saída: {self.output_dir.relative_to(BASE_DIR)}/{{Faculdade}}/{{Curso}}/", flush=True)
        print("=" * 75, flush=True)

        for faculdade in faculdades_alvo:
            cursos = banco_cursos.get(faculdade, [])
            if not cursos:
                continue

            cursos_com_sucesso = 0
            print(f"\n🏛️  INSTITUIÇÃO: {faculdade} (Buscando {limit_cursos_por_faculdade if limit_cursos_por_faculdade > 0 else len(cursos)} cursos com documentos)", flush=True)
            print("-" * 75, flush=True)

            for curso in cursos:
                if limit_cursos_por_faculdade > 0 and cursos_com_sucesso >= limit_cursos_por_faculdade:
                    break
                try:
                    baixados = self.coletar_curso(faculdade, curso, max_docs=max_docs_por_curso)
                    total_baixados_geral += baixados
                    if baixados > 0:
                        cursos_com_sucesso += 1
                except Exception as e:
                    print(f"   [!] Erro no curso {curso.get('curso', '')}: {e}", flush=True)

        tempo_total = time.time() - inicio
        print("\n" + "=" * 75, flush=True)
        print("🎉 PIPELINE DE WEB SCRAPING CONCLUÍDO COM SUCESSO!", flush=True)
        print(f"  • Total de documentos processados: {total_baixados_geral}", flush=True)
        print(f"  • Tempo de execução: {tempo_total:.2f}s", flush=True)
        print(f"  • Catálogo consolidado gerado em: {self.catalogo_path.relative_to(BASE_DIR)}", flush=True)
        print("=" * 75, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Web Scraper e Classificador do AcademicIA")
    parser.add_argument("--faculdade", type=str, default="todas", help="Instituição específica (UFPI, UESPI, IFPI, UFDPAR ou todas)")
    parser.add_argument("--limit-cursos", type=int, default=2, help="Limite de cursos por faculdade (0 para todos)")
    parser.add_argument("--max-docs", type=int, default=2, help="Máximo de documentos baixados por curso")
    parser.add_argument("--delay", type=float, default=0.3, help="Tempo de espera entre requisições em segundos")
    args = parser.parse_args()

    faculdades = None
    if args.faculdade.lower() != "todas":
        faculdades = [args.faculdade.upper()]

    scraper = WebScraperAcademicIA(delay=args.delay)
    scraper.executar(
        faculdades=faculdades,
        limit_cursos_por_faculdade=args.limit_cursos,
        max_docs_por_curso=args.max_docs
    )


if __name__ == "__main__":
    main()

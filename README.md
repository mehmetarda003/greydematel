# Grey DEMATEL (Python + Streamlit)

Gri sayılarla **DEMATEL** (Decision Making Trial and Evaluation Laboratory) analizi. Uzmanların kriterler arası etkileşimlerini aralık değerlerle modelleyip toplam ilişki matrisini, R/C toplamlarını, kriter ağırlıklarını ve etki-yönlü grafi üretir.

## Yöntem özeti

1. **A** — Direkt ilişki matrisi (köşegen `[0,0]`, dil skalası `[0,0]` … `[3,4]`).
2. **D** — `S = max(max_i Σ_j ā_ij, max_j Σ_i ā_ij)`; `D = A / S`.
3. **T** — `T⁻ = D⁻ (I − D⁻)⁻¹`, `T⁺ = D⁺ (I − D⁺)⁻¹`.
4. **R, C** — `T` durulaştırıldıktan sonra (`(T⁻+T⁺)/2`) satır/sütun toplamları; ağırlık `R+C` üzerinden normalize. Gri `R−C` referansı: alt = `R⁻−C⁺`, üst = `R⁺−C⁻`.
5. **Graf** — Eşik üstü ilişkilerle yönlü etki diyagramı; neden–sonuç diyagramı (`R−C` vs `R+C`).

## Kurulum

```bash
cd dematel
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
streamlit run app.py
```

Tarayıcıda matrisi skala ile doldurun veya alt/üst limit CSV veya Excel yükleyin.

## Proje yapısı

- `grey_dematel/core.py` — hesaplama
- `grey_dematel/scales.py` — dil skalası
- `app.py` — Streamlit arayüzü

## GitHub

```bash
git init
git add .
git commit -m "Grey DEMATEL Streamlit uygulaması"
gh repo create grey-dematel --public --source=. --push
```

(`gh` yüklü değilse GitHub’da boş repo oluşturup `git remote add origin ...` ile push edin.)

## Lisans

Proje sahibinin tercihine göre eklenebilir.

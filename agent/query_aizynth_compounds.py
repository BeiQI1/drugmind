import argparse
import csv
import json
import time
import urllib.parse
import urllib.request


def _iter_mol_nodes(node):
    if isinstance(node, dict):
        if node.get("type") == "mol":
            yield node
        children = node.get("children") or []
        for child in children:
            yield from _iter_mol_nodes(child)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_mol_nodes(item)


def extract_molecules_from_aizynth_output(data):
    mols = {}
    leaf_smiles = set()

    for tree in data or []:
        for mol in _iter_mol_nodes(tree):
            smiles = mol.get("smiles")
            if not smiles:
                continue

            in_stock = bool(mol.get("in_stock", False))
            if smiles in mols:
                mols[smiles]["in_stock"] = mols[smiles]["in_stock"] or in_stock
            else:
                mols[smiles] = {"smiles": smiles, "in_stock": in_stock}

            children = mol.get("children") or []
            if not children:
                leaf_smiles.add(smiles)

    for smiles, rec in mols.items():
        rec["is_leaf"] = smiles in leaf_smiles

    return mols


class PubChemClient:
    def __init__(self, timeout_s=20, delay_s=0.2):
        self.timeout_s = timeout_s
        self.delay_s = delay_s
        self._last_request_ts = 0.0
        self._cache = {}

    def _sleep_if_needed(self):
        now = time.time()
        elapsed = now - self._last_request_ts
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)
        self._last_request_ts = time.time()

    def _get_json(self, url):
        self._sleep_if_needed()
        req = urllib.request.Request(url, headers={"User-Agent": "drugtoolagent/1.0 (PubChem PUG-REST)"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def query_by_smiles(self, smiles, max_synonyms=20):
        if smiles in self._cache:
            return self._cache[smiles]

        q = urllib.parse.quote(smiles, safe="")
        out = {
            "cid": None,
            "iupac_name": None,
            "inchi_key": None,
            "canonical_smiles": None,
            "synonyms": [],
            "error": None,
        }

        try:
            prop_url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                f"{q}/property/IUPACName,InChIKey,CanonicalSMILES/JSON"
            )
            prop = self._get_json(prop_url)
            props = (prop.get("PropertyTable") or {}).get("Properties") or []
            if props:
                p0 = props[0]
                out["cid"] = p0.get("CID")
                out["iupac_name"] = p0.get("IUPACName")
                out["inchi_key"] = p0.get("InChIKey")
                out["canonical_smiles"] = p0.get("CanonicalSMILES")

            syn_url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                f"{q}/synonyms/JSON"
            )
            syn = self._get_json(syn_url)
            infos = (syn.get("InformationList") or {}).get("Information") or []
            if infos:
                syns = infos[0].get("Synonym") or []
                out["synonyms"] = list(syns)[:max_synonyms]
        except Exception as e:
            out["error"] = str(e)

        self._cache[smiles] = out
        return out


def pick_best_name(iupac_name, synonyms):
    candidates = []
    if iupac_name:
        candidates.append(iupac_name)
    candidates.extend(synonyms or [])

    for name in candidates:
        if not isinstance(name, str):
            continue
        s = name.strip()
        if not s:
            continue
        if len(s) > 80:
            continue
        return s

    return iupac_name or (synonyms[0] if synonyms else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to AiZynthFinder output.json")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=0, help="Max SMILES to query (0 = all)")
    parser.add_argument("--only-in-stock", action="store_true")
    parser.add_argument("--only-leaf", action="store_true")
    parser.add_argument("--max-synonyms", type=int, default=20)
    parser.add_argument("--no-pubchem", action="store_true")
    args = parser.parse_args()

    with open(args.json, "r") as f:
        data = json.load(f)

    mols = extract_molecules_from_aizynth_output(data)
    rows = list(mols.values())

    if args.only_in_stock:
        rows = [r for r in rows if r["in_stock"]]
    if args.only_leaf:
        rows = [r for r in rows if r["is_leaf"]]

    rows.sort(key=lambda r: (not r["in_stock"], not r["is_leaf"], r["smiles"]))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    pc = None if args.no_pubchem else PubChemClient()

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "smiles",
                "in_stock",
                "is_leaf",
                "pubchem_cid",
                "inchi_key",
                "best_name",
                "iupac_name",
                "synonyms",
                "error",
            ],
        )
        writer.writeheader()

        for r in rows:
            info = {"cid": None, "iupac_name": None, "inchi_key": None, "synonyms": [], "error": None}
            if pc is not None:
                info = pc.query_by_smiles(r["smiles"], max_synonyms=args.max_synonyms)
            best = pick_best_name(info.get("iupac_name"), info.get("synonyms") or [])

            writer.writerow(
                {
                    "smiles": r["smiles"],
                    "in_stock": int(bool(r["in_stock"])),
                    "is_leaf": int(bool(r["is_leaf"])),
                    "pubchem_cid": info.get("cid"),
                    "inchi_key": info.get("inchi_key"),
                    "best_name": best,
                    "iupac_name": info.get("iupac_name"),
                    "synonyms": "|".join(info.get("synonyms") or []),
                    "error": info.get("error"),
                }
            )

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

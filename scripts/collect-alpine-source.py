#!/usr/bin/env python3
"""Collect the complete corresponding source of Alpine packages into one tar.

The boot artifacts redistribute Alpine's linux-lts kernel, which is GPL-2.0, so
each release has to offer the exact source of the kernel it ships from the same
place as the binaries (GPL-2.0 section 3). That means Alpine's build recipe at the
commit the package was built from (APKBUILD, patches, kernel config) and every
upstream file the recipe names. Alpine's mirrors drop a package as soon as a newer
one lands, so the source is collected and published rather than linked to.

An APKBUILD is a shell script, and this runs in the release job next to its
signing identity, so a recipe is only ever read as text, never executed. The
recipe's sha512sums block says which files belong to it; a download is kept only
if it matches that checksum, so a wrong URL can fail the run but cannot change
what is published.

Usage:
    collect-alpine-source.py OUTPUT.tar [--extra DIR] ORIGIN@VERSION@APORTS_COMMIT...

Example:
    collect-alpine-source.py dist/src.tar linux-lts@6.18.63-r0@aabe3e5ce3...
"""

import argparse
import hashlib
import io
import re
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

# Alpine's own mirror of aports. Its GitLab refuses requests from CI runners.
APORTS = "https://codeload.github.com/alpinelinux/aports/tar.gz"
DISTFILES = "https://distfiles.alpinelinux.org/distfiles"
DISTFILES_BRANCHES = ["v3.24", "v3.23", "v3.22", "v3.21", "v3.20", "edge"]
USER_AGENT = "beskar7-inspector-source-collector"
ATTEMPTS = 4


def fetch(url, dest=None):
    """Download url to dest, or return its bytes. Returns None on a 404/410."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if dest is None:
                    return response.read()
                with open(dest, "wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                return dest
        except urllib.error.HTTPError as err:
            if err.code in (403, 404, 410):
                return None
            failure = err
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            failure = err
        if attempt < ATTEMPTS:
            time.sleep(attempt * 5)
    print(f"  giving up on {url}: {failure}", file=sys.stderr)
    return None


def sha512(path):
    digest = hashlib.sha512()
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_recipe(origin, commit, dest):
    """Unpack main/ or community/<origin> at commit into dest; return the repo name."""
    data = fetch(f"{APORTS}/{commit}")
    if data is None:
        raise SystemExit(f"could not fetch aports at commit {commit}")
    repo = None
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts[1:]
            if not member.isfile() or len(parts) < 3 or parts[0] not in ("main", "community") or parts[1] != origin:
                continue
            relative = PurePosixPath(*parts[2:])
            if ".." in relative.parts:
                raise SystemExit(f"unsafe path in recipe archive: {member.name}")
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
            repo = parts[0]
    if repo is None or not (dest / "APKBUILD").is_file():
        raise SystemExit(f"no recipe for {origin} at aports commit {commit}")
    return repo


def expand(text, variables):
    """Expand the simple parameter forms recipes use in version strings and URLs."""

    def replace(match):
        name, op, pattern = match.group("name") or match.group("bare"), match.group("op"), match.group("pat")
        value = variables.get(name, "")
        if op in ("%", "%%") and pattern and pattern.endswith("*"):
            literal = pattern[:-1]
            index = value.find(literal) if op == "%%" else value.rfind(literal)
            return value[:index] if index >= 0 else value
        if op in ("#", "##") and pattern and pattern.startswith("*"):
            literal = pattern[1:]
            index = value.rfind(literal) if op == "##" else value.find(literal)
            return value[index + len(literal):] if index >= 0 else value
        if op == "//" and pattern and "/" in pattern:
            old, new = pattern.split("/", 1)
            return value.replace(old, new)
        if op == "/" and pattern and "/" in pattern:
            old, new = pattern.split("/", 1)
            return value.replace(old, new, 1)
        return value

    form = re.compile(
        r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:(?P<op>%%|%|##|#|//|/)(?P<pat>[^}]*))?\}"
        r"|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"
    )
    return form.sub(replace, text)


def parse_recipe(text):
    """Read a recipe as text: its variables, checksums and candidate source URLs."""
    variables = {}
    for match in re.finditer(r'^([A-Za-z_][A-Za-z0-9_]*)=(?:"([^"\n]*)"|([^\s"\'(]*))\s*(?:#.*)?$', text, re.M):
        value = match.group(2) if match.group(2) is not None else match.group(3)
        variables.setdefault(match.group(1), expand(value, variables))

    block = re.search(r'^sha512sums="(.*?)"', text, re.M | re.S)
    if block is None:
        block = re.search(r"^sha512sums='(.*?)'", text, re.M | re.S)
    if block is None:
        raise SystemExit("recipe has no sha512sums block")
    checksums = {}
    for line in block.group(1).split("\n"):
        fields = line.split()
        if len(fields) == 2:
            checksums[fields[1]] = fields[0]

    urls = {}
    for token in re.findall(r'(?:[^\s"\']+::)?https?://[^\s"\']+', text):
        name, _, url = token.rpartition("::")
        url = expand(url, variables)
        urls.setdefault(expand(name, variables) or PurePosixPath(urllib.parse.urlparse(url).path).name, []).append(url)
    return variables, checksums, urls


def collect_package(spec, bundle):
    try:
        origin, version, commit = spec.split("@")
    except ValueError:
        raise SystemExit(f"bad package spec (want ORIGIN@VERSION@COMMIT): {spec}")

    print(f"{origin} {version} (aports {commit[:12]})", flush=True)
    package = bundle / f"{origin}-{version}"
    recipe_dir = package / "aports"
    distfiles = package / "distfiles"
    recipe_dir.mkdir(parents=True)
    distfiles.mkdir()

    repo = fetch_recipe(origin, commit, recipe_dir)
    variables, checksums, urls = parse_recipe((recipe_dir / "APKBUILD").read_text())

    # The recipe must be the one that produced this exact package.
    built = f"{variables.get('pkgver')}-r{variables.get('pkgrel')}"
    if built != version:
        raise SystemExit(f"{origin}: the recipe at {commit} builds {built}, not {version}")

    for name, expected in sorted(checksums.items()):
        local = recipe_dir / name
        if local.is_file():
            if sha512(local) != expected:
                raise SystemExit(f"{origin}: {name} in the recipe does not match its checksum")
            continue
        target = distfiles / name
        candidates = urls.get(name, []) + [f"{DISTFILES}/{branch}/{name}" for branch in DISTFILES_BRANCHES]
        for url in candidates:
            if fetch(url, target) and sha512(target) == expected:
                print(f"  {name} ({target.stat().st_size / 1048576:.1f} MB) from {url}", flush=True)
                break
            target.unlink(missing_ok=True)
        else:
            raise SystemExit(f"{origin}: could not fetch {name} with the checksum its recipe names")
    return f"  {origin} {version}  (aports {repo}/{origin} at {commit})"


def add_tree(archive, root):
    def clean(info):
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        info.mtime = 0
        info.mode = 0o755 if info.isdir() else 0o644
        return info

    for path in sorted(root.rglob("*")):
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False, filter=clean)


def main():
    parser = argparse.ArgumentParser(description="Collect the source of Alpine packages into one tar.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--extra", type=Path, help="directory copied into the bundle as inspector-build/")
    parser.add_argument("packages", nargs="+", metavar="ORIGIN@VERSION@APORTS_COMMIT")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as work:
        bundle = Path(work, "bundle")
        bundle.mkdir()
        listing = [collect_package(spec, bundle) for spec in args.packages]
        readme = [
            "Complete corresponding source of the Alpine packages listed below, as redistributed",
            "by beskar7-inspector (https://github.com/beskar7/beskar7-inspector).",
            "",
            "Each <package>-<version>/ directory holds:",
            "  aports/     Alpine's build recipe at the commit named below: the APKBUILD, its",
            "              patches and, for the kernel, its configuration",
            "  distfiles/  the upstream files the recipe names, verified against the recipe's",
            "              sha512 checksums",
            "",
            "To rebuild a package on the matching Alpine release:",
            "  cd <package>-<version>/aports && SRCDEST=$PWD/../distfiles abuild -r",
            "",
        ]
        if args.extra:
            readme += [
                "inspector-build/ holds the files that select what the inspector ships from these",
                "packages and assemble it into the boot artifacts.",
                "",
            ]
            for path in sorted(args.extra.rglob("*")):
                if path.is_file():
                    target = bundle / "inspector-build" / path.relative_to(args.extra)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
        (bundle / "README.txt").write_text("\n".join(readme + ["Packages:"] + listing) + "\n")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w", format=tarfile.GNU_FORMAT) as archive:
            add_tree(archive, bundle)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()

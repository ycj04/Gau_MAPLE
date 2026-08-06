# Gau_MAPLE v0.10.0

Experimental Gaussian `External` interface for MAPLE calculators. Gau_MAPLE
translates Gaussian `.EIn/.EOut` requests and forwards energies, Cartesian
gradients, and optional Hessians to persistent MAPLE calculator servers.
Gaussian remains responsible for optimization, constraints, transition-state
searches, numerical frequencies, and IRC propagation.

## Scope

This is interface software, not a chemical-accuracy benchmark. Normal Gaussian
termination proves that the protocol completed; it does not establish that a
model is accurate for a particular element set, charge, spin state, reaction,
or geometry. Validate every selected checkpoint for the intended chemistry.

Gaussian, MAPLE, model runtimes, plugins, and checkpoints are not distributed
with this repository.

### Clone the repository

```bash
git clone https://github.com/ycj04/Gau_MAPLE.git
cd Gau_MAPLE
```

### Verify Gaussian and MAPLE first

Confirm Gaussian works independently and that MAPLE imports in each Python
environment that will host a server:

```bash
which g16
python - <<'PY'
import maple
print(maple.__file__)
PY
```

A typical installation uses one MAPLE environment for AIMNet2/ANI/MACE and a
second environment for UMA/eSEN. A single environment is also valid when all
selected calculators coexist there.

## Installation

Install Gau_MAPLE into the environment that runs `maple_server`:

```bash
python -m pip install -e '.[test]'
pytest -q
```

For a second UMA/eSEN environment, install the same checkout there as well:

```bash
/path/to/meta-env/bin/python -m pip install -e .
```

## Configuration

Copy the template and replace every placeholder absolute path:

```bash
cp config/profiles.example.toml config/profiles.toml
```

For a one-profile installation, keep only the required profile and server blocks.

Set the shared configuration path in shells that launch servers or Gaussian:

```bash
export GAU_MAPLE_CONFIG="$PWD/config/profiles.toml"
```

`profiles.example.toml` shows the validated two-server layout:

```text
Gaussian -> gau-maple client
             |-- maple_server: AIMNet2, ANI, MACE
             `-- meta_server:  UMA-S, eSEN
```

The `maceomol_native` and `macepolm_native` profiles require the separately
installed `maple_mace_native` plugin. Gau_MAPLE does not bundle that plugin or
its model files.

## Implemented interface capabilities

| Capability | Status | Notes |
| --- | --- | --- |
| Single-point energy | Implemented | Gaussian External derivative level 0. |
| Geometry optimization | Implemented | Gaussian consumes External energy and gradients. |
| Constraints / ModRedundant | Interface-supported | Gaussian controls constraints. |
| Relaxed Scan | Interface-supported | Gaussian controls the scan; the backend supplies derivatives. |
| Frequency | Implemented | Analytic Hessian when the profile provides one; otherwise use `Freq=Numer`. |
| Transition-state optimization | Interface-supported | Confirm exactly one imaginary mode. |
| IRC | Interface-supported | Start from a validated first-order saddle point. |
| Charge and multiplicity policy | Implemented | Independent per-profile fail-fast rules. |
| Persistent model servers | Implemented | Unix-domain sockets with profile preloading. |
| Cross-environment routing | Implemented | Different profiles may run in different Conda environments. |
| IR/Raman intensities | Not implemented | Dipole/polarizability derivatives are not supplied. |
| Generic solvent/D4 pass-through | Not implemented | Requires explicit support in each MAPLE calculator/profile. |
| PBC / MM embedding | Not implemented | Unsupported requests fail explicitly. |

## Electronic-state policy

The distributed example configuration uses the following validated engineering
contract. This describes parameter propagation, not chemical accuracy.

| Profile | Charge | Multiplicity |
| --- | --- | --- |
| `aimnet2` | supported | singlet only |
| `aimnet2nse` | supported | supported |
| `ani1x`, `ani1ccx`, `ani1xnr`, `ani2x` | neutral only | singlet only |
| `maceoff23m` | neutral only | singlet only |
| `maceomol_native` | supported | supported |
| `macepolm_native` | supported | supported |
| `uma-s-1p2` with OMol task | supported | supported |
| `esen-sm-conserving-all` OMol checkpoint | supported | supported |

For the native MACE-Polar profile, Gaussian multiplicity is mapped directly to
MACE `spin`: singlet=1, doublet=2, triplet=3.

## Server mode

Validate, start, inspect, and stop the configured servers:

```bash
gau-maple-ctl --config config/profiles.toml validate
gau-maple-ctl --config config/profiles.toml start all
gau-maple-ctl --config config/profiles.toml status all
gau-maple-ctl --config config/profiles.toml logs maple_server --lines 80
gau-maple-ctl --config config/profiles.toml stop all
```

Servers handle requests serially per process and retain loaded calculator
objects between calls. Use separate servers when incompatible runtimes cannot
coexist in one environment.

## Gaussian External entry point

Use absolute paths inside Gaussian route sections. Minimal AIMNet2 single point:

```text
%chk=water.chk
#p external='/absolute/path/to/maple-env/bin/gau-maple --config /absolute/path/to/Gau_MAPLE/config/profiles.toml --profile aimnet2'

Gau_MAPLE water single point

0 1
O  0.000000  0.000000  0.000000
H  0.757160  0.000000  0.586260
H -0.757160  0.000000  0.586260
```

Run normally:

```bash
g16 < job.gjf > job.log
```

Optimization and frequency examples:

```text
#p opt=nomicro external='... gau-maple --config ... --profile aimnet2'
#p freq external='... gau-maple --config ... --profile aimnet2'
```

For profiles without an analytic Hessian, use Gaussian numerical frequencies:

```text
#p freq=numer external='... gau-maple --config ... --profile macepolm_native'
```

Additional inputs are in [`examples/`](examples/).

## Diagnostics and validation

```bash
pytest -q
gau-maple-ctl --config config/profiles.toml validate
gau-maple-doctor --config config/profiles.toml --gaussian "$(command -v g16)"
gau-maple-validation --config config/profiles.toml capabilities
gau-maple-validation --config config/profiles.toml stability
```

For native MACE profiles, also confirm that the separately installed plugin and
checkpoint paths resolve in the Python environment hosting `maple_server`.

## Configuration reference

Profile fields:

| Field | Purpose |
| --- | --- |
| `model` | MAPLE calculator registry name. |
| `device` | Calculator device, commonly `cpu` or `cuda`. |
| `charge_policy` | `supported`, `neutral_only`, or `calculator`. |
| `multiplicity_policy` | `supported`, `singlet_only`, or `calculator`. |
| `strict_charge_multiplicity` | Require explicit backend handling. |
| `reject_mm_charges` | Reject unsupported Gaussian MM point charges. |
| `model_options` | Calculator-specific constructor options. |

Server fields define the environment-specific `gau-maple-server` executable,
profile list, Unix socket, PID/log paths, preload behavior, and timeouts. See
[`config/profiles.example.toml`](config/profiles.example.toml).

## Troubleshooting

- Run `gau-maple-ctl --config config/profiles.toml validate` before starting servers.
- Use `gau-maple-ctl --config config/profiles.toml logs SERVER --lines 100` for startup failures.
- Confirm each server executable belongs to the intended Conda environment.
- Use absolute paths in Gaussian `External` route sections.

## Scientific and protocol limits

- MAPLE forces are converted to Gaussian gradients with the required sign and
  Bohr/angstrom conversion.
- A model accepting charge or multiplicity metadata does not prove open-shell or
  ionic chemical accuracy.
- Gaussian controls ModRedundant constraints, relaxed scans, TS searches, and
  IRC; Gau_MAPLE supplies the requested potential-energy derivatives.
- Generic external-force potentials, solvent corrections, and D4 corrections
  are not added automatically by Gaussian External. They require calculator-
  specific support in MAPLE and explicit profile configuration.
- Model files and third-party runtimes remain subject to their own licenses.

## License

Gau_MAPLE is released under the MIT License. Gaussian, MAPLE, MLIP runtimes,
plugins, and checkpoints are separate third-party software and are not
relicensed or redistributed here.

## Acknowledgments

Gau_MAPLE uses Gaussian's public External protocol and the calculator
interface exposed by [MAPLE](https://github.com/ClickFF/MAPLE). MAPLE is an
independent third-party project: it is not bundled, redistributed, or
relicensed by Gau_MAPLE. Users are responsible for complying with MAPLE's
current upstream terms and with the licenses of any selected MLIP runtime,
plugin, or checkpoint. The project was developed with architectural
inspiration from Gau_UMA and Gau_Skala.

The author ([ycj04](https://github.com/ycj04)) experienced anxiety symptoms
and hoped that working through this project would be a constructive step toward
regaining stability. Please feel free to get in touch with any ideas,
suggestions, or concerns that this project may infringe on your rights. All
faults are mine.

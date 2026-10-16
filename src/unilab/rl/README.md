# In-tree training implementation

UniLab owns this package, including algorithms, collectors, IPC and training
logging. Environment construction stays behind `env_contract.EnvFactory` so
spawn collectors do not depend on task registrations in the parent process.

The initial generic implementation was brought back from the Apache-2.0
`unilab-rl==1.0.0` source distribution (SHA256
`b2ab788a99054b2a75183b08c76c9cb5e70af5a6ed749316106a082ded9fbe16`).
Its Python modules were moved from `uni_rl` to `unilab.rl`; this repository now
maintains them. Upstream source: https://github.com/unilabsim/unilab_rl/tree/ce0050ecc77ec17de84b6b581148c67e1ae03caa.
The repository's Apache-2.0 LICENSE applies; upstream copyright notices are
preserved. Namespace relocation alone does not change algorithm defaults.

AMP and generic distillation were not present in that published source. Their
implementations are restored from the Apache-2.0 source at
https://github.com/Andisyc/ReGait/tree/06c12ee605795daa14817c9db7644ba75a453075/src/unilab/algos/torch,
then adapted to this checkout's injected data, EnvFactory and runtime contracts.
No runtime imports fall back to an installed external training package.

The namespace move preserves the generic implementation baseline. AMP and
distillation adaptation require their own behavioral checks; source restoration
alone does not prove training success. The optional AMP sensor-history backend
capability remains unavailable in the pinned published UniSim backend; the facade
reports that limitation explicitly instead of fabricating history samples.

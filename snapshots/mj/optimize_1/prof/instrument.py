# Generates an instrumented copy of mj.mojo with per-phase wall-clock timers (perf_counter_ns).
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
s = s.replace("from std.math import", "from std.time import perf_counter_ns\nfrom std.math import", 1)
s = s.replace("    var rope_sin: FPtr\n", "    var rope_sin: FPtr\n    var prof: IPtr\n", 1)
s = s.replace("        self.rope_sin = falloc(half)\n", "        self.rope_sin = falloc(half)\n        self.prof = ialloc(16)\n", 1)
L = s.split('\n'); out = []
cats = [('matmul(self.q,', 0, 3), ('matmul(self.xb2', 1, 1), ('matmul(self.hb,', 2, 2), ('matmul(self.xb, self.hb', 3, 1), ('matmul(self.logits', 4, 1),
        ('# multihead attention', 5, None)]
i = 0
while i < len(L):
    line = L[i]; hit = None
    for pat, k, n in cats:
        if pat in line: hit = (k, n)
    if hit and hit[1] is not None:
        ind = line[:len(line) - len(line.lstrip())]
        out.append(f'{ind}var _t{i} = perf_counter_ns()')
        out.extend(L[i:i + hit[1]])
        out.append(f'{ind}self.prof[unsafe_offset={hit[0]}] += Int(perf_counter_ns() - _t{i})')
        i += hit[1]; continue
    if hit and hit[1] is None:  # attention block: until the wo matmul
        ind = line[:len(line) - len(line.lstrip())]
        out.append(f'{ind}var _ta = perf_counter_ns()')
        j = i
        while 'matmul(self.xb2' not in L[j]: out.append(L[j]); j += 1
        out.append(f'{ind}self.prof[unsafe_offset=5] += Int(perf_counter_ns() - _ta)')
        i = j; continue
    out.append(line); i += 1
s = '\n'.join(out)
s = s.replace("    def forward(mut self, token: Int, pos: Int):\n", "    def forward(mut self, token: Int, pos: Int):\n        var _tf = perf_counter_ns()\n        self.forward_(token, pos)\n        self.prof[unsafe_offset=6] += Int(perf_counter_ns() - _tf)\n        self.prof[unsafe_offset=7] += 1\n\n    def forward_(mut self, token: Int, pos: Int):\n", 1)
s = s.replace('''        if starts_with(line, "QUIT"):
            return''', '''        if starts_with(line, "QUIT"):
            var names = ["qkv", "wo", "w1+w3", "w2", "classifier", "rope+attention", "forward total"]
            var tot = Float64(model.prof[unsafe_offset=6])
            var ntok = model.prof[unsafe_offset=7]
            for c in range(7):
                var ns = Float64(model.prof[unsafe_offset=c])
                print("PROF", names[c], "ms/token", ns / Float64(ntok) / 1e6, "share", ns / tot, file=RawFileDescriptor(2))
            print("PROF tokens", ntok, file=RawFileDescriptor(2))
            return''', 1)
open(dst, 'w').write(s)

"""Talks to one implementation process. All timestamps are taken here, outside the program, with a monotonic clock."""
import os, subprocess, time

CLEAN = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS']


class Proc:
    def __init__(self, cmd, model, tokenizer, env=None, cwd=None, timeout=600):
        e = {'PATH': '/usr/bin:/bin', 'HOME': os.environ.get('HOME', '/tmp'), **{k: '1' for k in CLEAN}, **(env or {})}
        self.t_launch = time.perf_counter_ns()
        self.p = subprocess.Popen(list(cmd) + [model, tokenizer], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env=e, cwd=cwd, text=True, bufsize=1)
        self.timeout = timeout
        line = self.read()
        if line != 'READY':
            raise RuntimeError(f'no READY: {line!r}')
        self.t_ready = time.perf_counter_ns()

    def read(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f'closed early, exit {self.p.poll()}, stderr {self.p.stderr.read()[:300]!r}')
        return line.rstrip('\n')

    def send(self, s):
        self.p.stdin.write(s + '\n')
        self.p.stdin.flush()

    def ask(self, s):
        self.send(s)
        return self.read()

    def generate(self, max_new, text):
        """Returns ids, reason, ns timestamps of each token, ns at send, ns at DONE."""
        t0 = time.perf_counter_ns()
        self.send(f'GENERATE {max_new} {text}')
        ids, stamps = [], []
        while True:
            line = self.read()
            now = time.perf_counter_ns()
            if line.startswith('T '):
                ids.append(int(line[2:]))
                stamps.append(now)
            elif line.startswith('DONE '):
                _, n, why = line.split()
                if int(n) != len(ids):
                    raise RuntimeError(f'count {n} != {len(ids)}')
                return ids, why, stamps, t0, now
            else:
                raise RuntimeError(f'bad line {line[:80]!r}')

    def fixed(self, ids):
        t0 = time.perf_counter_ns()
        p = self.ask('FIXED ' + ' '.join(map(str, ids))).split()
        t1 = time.perf_counter_ns()
        if p[0] != 'DONE' or int(p[1]) != len(ids):
            raise RuntimeError(f'bad FIXED answer {p[:3]}')
        return list(map(int, p[2:])), t1 - t0

    def close(self):
        """Returns exit status and peak resident bytes for this process (macOS reports ru_maxrss in bytes)."""
        try:
            self.send('QUIT')
        except BrokenPipeError:
            pass
        self.p.stdin.close()
        _, status, ru = os.wait4(self.p.pid, 0)
        self.p.returncode = os.waitstatus_to_exitcode(status)
        self.cpu_s = ru.ru_utime + ru.ru_stime
        self.wall_s = (time.perf_counter_ns() - self.t_launch) / 1e9
        return self.p.returncode, ru.ru_maxrss

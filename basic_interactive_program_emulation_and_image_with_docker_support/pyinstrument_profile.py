import pyinstrument
prof = pyinstrument.Profiler()
prof.start()
... # long running program. remember to timeout
prof.stop()
prof.print()
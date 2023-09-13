import pstats

# from pstats import SortKey
p = pstats.Stats("alpine_actor.profile")
stats = p.strip_dirs().sort_stats(2)
stats.print_stats()

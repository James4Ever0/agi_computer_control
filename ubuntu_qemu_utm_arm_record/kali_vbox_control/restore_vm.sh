# take snapshot
# vboxmanage snapshot "Ubuntu 16.04" take "Snapshot 3"
# restore snapshot
vboxmanage snapshot "Ubuntu 16.04" restore "Snapshot 14"
# 12 has bug.
# vboxmanage snapshot "Ubuntu 16.04" restore "Snapshot 7"
# vboxmanage snapshot "Ubuntu 16.04" restore "Snapshot 6"
# vboxmanage snapshot "Ubuntu 16.04" restore "Snapshot 3"
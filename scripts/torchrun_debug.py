#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from torch.distributed.run import main
import debugpy
import os
if __name__ == '__main__':

    # wait for debugger attach
    print("Waiting for debugger attach...")
    debugpy.listen(("127.0.0.1", 5678))
    debugpy.wait_for_client()
    debugpy.breakpoint()

    # "TORCHINDUCTOR_COMPILE_THREADS": "2"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "2"
    print("Debugger attached.")

    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())

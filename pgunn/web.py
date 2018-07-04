import os
import datetime
import time

while True:
    os.system("ssh xbucha02@merlin.fit.vutbr.cz '(cd WWW ; python3.6 get_data.py)' > results.out")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M"))
    time.sleep(300)

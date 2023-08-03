#!/bin/sh
declare -a StringArray=(Linux Mint Fedora Red Ubuntu Debian )

for file in ${StringArray[@]}
do
    echo python test_audio.py --file_name $file.txt --epoch_name generator-80.pkl
done


To make it work 
1- sed -i 's/\r//' Ne.sh
2- Bash it


#!/bin/sh
declare -a StringArray=(2BqVo8kVB2Skwgyb   8B9N9jOOXGUordVG   D4jGxZ7KamfVo4E2V  jgxq52DoPpsR9ZRx   neaPN7GbBEUex8rV    vnljypgejkINbBAY  ywE435j4gVizvw3R
2ojo7YRL7Gck83Z3   8e5qRjN7dGuovkRY   DMApMRmGq5hjkyvX   k5bqyxx2lzIbrlg9   NgQEvO2x7Vh3xy2xz  Pz327QrLaGuxW8Do         W4XOzzNEbrtZz4dW  Z7BXPnplxGUjZdmBZ
35v28XaVEns4WXOv   9EWlVBQo9rtqRYdy   DWmlpyg93YCXAXgE   KLa5k73rZvSlv82X   NgXwdx5KkZI5GRWa   Q4vMvpXkXBsqryvZ         W7LeKXje7QhZlLKe  zaEBPeMY4NUbDnZy
4aGjX3AG5xcxeL7a   9Gmnwa5W9PIwaoKq   DWNjK4kYDACjeEg3   kNnmb7MdArswxLYw   nO2pPlZzv3IvOQoP2  qNY4Qwveojc8jlm4         wa3mwLV3ldIqnGnV  Ze7YenyZvxiB4MYZ
4BrX8aDqK2cLZRYl   9MX3AgZzVgCw4W4j   eBQAWmMg4gsLYLLa   KqDyvgWm4Wh8ZDM7   NWAAAQQZDXC5b9Mk   R3mexpM2YAtdPbL7         WYmlNV2rDkSaALOE  ZebMRl5Z7dhrPKRD
52XVOeXMXYuaElyw   9mYN2zmq7aTw4Blo   EExgNZ9dvgTE3928   kxgXN97ALmHbaezp   ObdQbr9wyDfbmW4E   R3mXwwoaX9IoRVKe         X4vEl3glp9urv4GN  zwKdl7Z2VRudGj2L
5BEzPgPKe8taG9OB   anvKyBjB5OiP5dYZ   eL2w4ZBD7liA85wm   ldrknAmwYPcWzp4N   OepoQ9jWQztn5ZqL   RjDBre8jzzhdr4YL         xEYa2wgAQof3wyEO  zZezMeg5XvcbRdg3
5o9BvRGEGvhaeBwA   aokxBz9LxXHzZzay   eLQ3mNg27GHLkDej   LR5vdbQgp3tlMBzB   oNOZxyvRe3Ikx3La   ro5AaKwypZIqNEp2         xPZw23VxroC3N34k
5pa4DVyvN2fXpepb   AvR9dePW88IynbaE   g2dnA9Wpvzi2WAmZ   M4ybygBlWqImBn9oZ  oOK5kxoW7dskMbaK   roOVZm7kYzS5d4q3         xRQE5VD7rRHVdyvM
73bEEYMKLwtmVwV43  AY5e3mMgZkIyG3Ox   G3QxQd7qGRuXAZda   mj4BWeRbp7ildyB9d  oRrwPDNPlAieQr8Q   Rq9EA8dEeZcEwada2        xwpvGaaWl5c3G5N3
7B4XmNppyrCK977p   BvyakyrDmQfWEABb   gNYvkbx3gof2Y3V9   mor8vDGkaOHzLLWBp  oXjpaOq4wVUezb3x   rwqzgZjbPaf5dmbL         xwzgmmv5ZOiVaxXz
7NEaXjeLX3sg3yDB   d2waAp3pEjiWgrDEY  gvKeNY2D3Rs2jRdL   mzgVQ4Z5WvHqgNmY   ppymZZDb2Bf4NQnE   V4ejqNL4xbUKkYrV         Xygv5loxdZtrywr9
7NqqnAOPVVSKnxyv   d3erpmyk9yFlVyrZ   Gym5dABePPHA8mZK9  n5XllaB4gZFwZXkBz  ppzZqYxGkESMdA5Az  V4ZbwLm9G5irobWn         YbmvamEWQ8faDPx2 )

for file in ${StringArray[@]}
do
     echo python evaluation.py -t /data/mohamed/data_emb2/txt/$file.txt -O /data/mohamed/data_emb2/enhanced/wavs/speakers/$file/ -M  /data/mohamed/data_emb2/best_frWavLMBase+.pt 
done
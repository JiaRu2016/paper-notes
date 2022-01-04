
[youtebe](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ) 
slides under each video

### Introduction

六种任务

- autio -> text. speech recognization. 语音识别  **It's not the seq2seq you know!**
-  autio -> audio. 
    + Speech Separation 从背景音中分离出语音、两个人说话分离
    + Voice Conversion 把一个人的声音变成柯南or新垣结衣. 难点：one-shot
- autio -> class.
    + which speaker?
    + keyword spotting.
        * Wake up words **需要模型非常轻量**
- text -> audio. Text to speech synthesis 语音合成
- text -> text. Translation, Summarization, Chat-bot, QA
- text -> class.

”硬train一发“的问题：label非常昂贵或不可得


### Speech Recognization (ASR)

#### input output

input: audio `(T_audio, D)`
output: sequence of tokens `(T_token, )`, token vocab size = `V`

输出形式有哪些选择？

- phoneme 发音最小单位，“音素”
    + Need **Lexicon** (mapping of word to seq of phoneme, languagist expert knowledge), which is expensive for some language
    + 声音和音素是一一对应的，比较好学
- grapheme 书写最小单位。
    + 英文：26 characters + space；中文 V=4000+
    + 声音和书写单位不是一一对应的，例如英文中发音'k'可以是k或c，中文的多音字。需要上下文信息， maybe languge model?
- morpheme: smallest meaningful unit, eg. "un, break, able", "re, kill, able"
    + 可以用语言学知识，也可以从大量语料库中用统计方法找pattern
- word: 中文 V=??
- byte: utf8 encoding. Language independent!

输入声学特征 `shape=(T, D)`

滑动窗口 eg. 10ms, 25ms => 100 frames ie. T=100 per second

waveform -(DFT)-> spectgram -> FilterBank -> log -> MFCC

- 原始声音 sample points
- log filter bank outputs `D=80`
- MFCC, `D=39`


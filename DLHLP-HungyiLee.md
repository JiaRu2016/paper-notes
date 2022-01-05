
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

#### eval metric

- LAS: WER word error rate: = (S + I + D) / N
- CTC: LER label error rate: edit distance ED(p, q) = number of insertion, substitution, and deletion requied to change p to q

#### model: LAS, typical seq2seq with attention

*Listen, Attend, and Spell (LAS)*

$$
\bold h = Listen(\bold x) \\
y_t = AttendAndSpell(\bold h, y_{<t}) \\
$$

Listen: RNN (possiblely with pooling or down-sampling) or TransformerEncoder. pyrimid-BLSTM

AttendAndSpell: 
$$
s_t = RNN(s_{t-1}, c_{t-1}, y_{t-1}) \\
c_t = AttentionContext(\bold h, s_t) \\
y_t = f(c_t, s_t)
$$

Beam search. hyperparam beam_width B, prune the V ary tree to make number of leaf always B.

Teacher Forcing: 如果用 $\hat y_{t-1}$ 当做RNN输入，那实际上在t>1位置上给了RNN非常多错误的训练样本。 应该用 ground truth $y_{t-1}$

关于Attention的讨论：Translation任务确实需要attention, 因为src和target位置对应关系不一定是顺序的，src最后一个单词可能对应target第一个单词；而语音识别任务则不是，位置对应关系应该是顺序的，所以有 location-aware attention

Language Model rescoring (over top 32 beam)
$$
s(\bold y|\bold x) = 
    \frac{\log P(\bold y|\bold x)}{|\bold y|_c} + 
    \lambda \log P_{LM}(\bold y|\bold x)
$$
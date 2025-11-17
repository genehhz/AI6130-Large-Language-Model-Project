# AI6130---Large-Language-Model---Project-

Project Assignment Submission for module AI6130, Large Language Model

Lin Yue (LINY0145@e.ntu.edu.sg), Jiang Kexun (kexun001@e.ntu.edu.sg), Ma Hong (HONG032@e.ntu.edu.sg), Pan Feng (Feng006@e.ntu.edu.sg), Salinelat Teixayavong (salinela.001@e.ntu.edu.sg), Eugene Ho (EHO010@e.ntu.edu.sg)

## Guba Emotion

The Chinese A-share market presents a unique case for financial analysis. Unlike mature markets, it is characterized by a high concentration of retail investors whose trading is often sentiment-driven, resulting in high volatility and frequent deviations from fundamental value. Consequently, traditional forecasting models rooted in the Efficient Market Hypothesis (EMH) are ill-equipped to handle these behavioral dynamics. To address the limitation, we developed a multi-modal prediction model that integrates social media sentiment with traditional financial market indicators. Our
framework analyzes textual data from the Eastmoney Stock Forum using a FinBERT-Tone-Chinese model, which we
fine-tuned on a domain-specific financial corpus. This sentiment analysis is then combined with technical market data to
forecast stock movements days ahead.

Our empirical analysis confirms two key findings. First, the domain-specific FinBERT-Tone-Chinese model substantially surpasses the accuracy of general-purpose financial BERT models in sentiment classification. Secondly, while all models performed comparably in aggregate, LightGBM, which leverages engineered features, achieved the best
individual-stock results, substantially outperforming transformer based models in the low signal-to-noise environment.
In summary, this research confirms that incorporating behavioral sentiment data with financial indicators is an effective
strategy to improve predictive accuracy in emerging markets driven by sentiment.

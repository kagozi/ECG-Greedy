flowchart LR
    %% --- Inputs ---
    R["<img src='https://raw.githubusercontent.com/kagozi/ECG-Greedy/main/rawsignal.png' width='70px' ><br>Raw ECGs<br>(12 × 1000)"]
    SCL["<img src='https://raw.githubusercontent.com/kagozi/ECG-Greedy/main/scalogram.png' width='70px' ><br>Scalograms<br>(224 × 224 × 12)"]
    PHS["<img src='https://raw.githubusercontent.com/kagozi/ECG-Greedy/main/phasogram.png' width='70px' ><br>Phasograms<br>(224 × 224 × 12)"]

    %% --- Transform ---
    R -.->|CWT| SCL
    R -.->|Phase| PHS

    %% --- Early Fusion: Channel Concat ---
    SCL --> EC["Early Concat<br>[SCL; PHS] ∈ ℝ^(B×24×224×224)"]
    PHS --> EC

    %% --- Adapter (24 → 3 ch) ---
    EC --> A["Adapter 1×1 Conv<br>24 → 3 ch"]

    %% --- Pretrained Backbone ---
    A --> PT["Pretrained Backbone<br>(e.g., EfficientNet)<br>→ f ∈ ℝ^(B×d)"]

    %% --- Classifier ---
    PT --> CLS["Classifier<br>d → 512 → 5<br>ReLU → BN → Drop"]

    %% --- Output ---
    subgraph Output ["Final Multilabel Classifier"]
        direction TB
        CLS --> O1(( ))
        CLS --> O2(( ))
        CLS --> O3(( ))
        CLS --> O4(( ))
        CLS --> O5(( ))
    end

    Output --> SIG["Sigmoid Focal Loss<br>(γ=2, α=0.25)"]

    %% --- Styling ---
    classDef img fill:#ffffff,stroke:#cccccc,stroke-width:1px
    classDef mod fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    classDef join fill:#ede7f6,stroke:#5e35b1,stroke-width:1px
    classDef head fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    classDef neuron fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px

    class R,SCL,PHS img
    class EC,A,PT mod
    class CLS,SIG head
    class O1,O2,O3,O4,O5 neuron

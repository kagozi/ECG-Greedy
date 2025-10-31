flowchart LR
    %% --- Inputs ---
    R["<img src='https://raw.githubusercontent.com/kagozi/ECG-Greedy/main/rawsignal.png' width='70px' ><br>Raw ECGs<br>(12 × 1000)"]
    SCL["<img src='https://raw.githubusercontent.com/kagozi/ECG-Greedy/main/scalogram.png' width='70px' ><br>Scalograms<br>(224 × 224 × 12)"]
    PHS["<img src='https://raw.githubusercontent.com/kagozi/ECG-Greedy/main/phasogram.png' width='70px' ><br>Phasograms<br>(224 × 224 × 12)"]

    %% --- Transform Arrows (optional, dotted for generation) ---
    R -.->|CWT| SCL
    R -.->|Phase| PHS

    %% --- Dual Stream CNNs ---
    SCL --> E1["Dual Stream CNN → fₛ ∈ ℝ^(B×d)"]
    PHS --> E2["Dual Stream CNN → fₚ ∈ ℝ^(B×d)"]

    %% --- Late Fusion ---
    E1 --> C["Concat [fₛ; fₚ] ∈ ℝ^(B×2d)"]
    E2 --> C

    %% --- Fusion + Classifier ---
    C --> FUS["Fusion MLP<br>2d → 1024 → ReLU → BN → Drop"]
    FUS --> CLS["Classifier<br>1024 → 512 → 5<br>ReLU → BN → Drop"]

    %% --- Output Layer ---
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
    classDef join fill: #ede7f6,stroke:#5e35b1,stroke-width:1px
    classDef head fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    classDef neuron fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px

    class R,SCL,PHS img
    class E1,E2 mod
    class C join
    class FUS,CLS,SIG head
    class O1,O2,O3,O4,O5 neuron

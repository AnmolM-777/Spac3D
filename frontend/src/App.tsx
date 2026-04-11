import React, { useState, useCallback, useRef } from 'react'

interface Detection {
  label: string
  confidence: number
  bbox: [number, number, number, number]
}

interface FurnitureItem {
  id: string
  name: string
  style: string
  image_url: string
  similarity_score: number
}

interface AnalysisResult {
  status: string
  detections: Detection[]
  style: string
  confidence: number
  recommendations: FurnitureItem[]
  image_size: { width: number; height: number }
}

const STYLE_ACCENTS: Record<string, string> = {
  Minimalist: '#A8DADC',
  Modern: '#E9C46A',
  Bohemian: '#E76F51',
  Industrial: '#8D99AE',
  Scandinavian: '#95D5B2',
  Traditional: '#C77DFF',
}

const API_URL = 'http://127.0.0.1:8000'

export default function App() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const CATEGORY_MAP: Record<string, string[]> = {
    chair: ['chair', 'armchair', 'seat', 'pouf', 'stool'],
    couch: ['sofa', 'couch', 'sectional'],
    sofa: ['sofa', 'couch', 'sectional'],
    bed: ['bed', 'mattress'],
    'dining table': ['table', 'desk', 'console'],
    table: ['table', 'desk', 'console'],
    'potted plant': ['plant', 'planter'],
    plant: ['plant', 'planter'],
    tv: ['tv', 'console', 'media', 'stand'],
    television: ['tv', 'console', 'media', 'stand'],
    refrigerator: ['fridge', 'refrigerator'],
    clock: ['clock'],
    vase: ['vase'],
    book: ['book', 'shelf', 'bookcase'],
    lamp: ['lamp', 'light', 'pendant', 'chandelier', 'sconce'],
    laptop: ['desk', 'office'],
    monitor: ['desk', 'console'],
    bench: ['bench', 'stool', 'ottoman'],
    desk: ['desk', 'table'],
    cabinet: ['cabinet', 'sideboard', 'credenza', 'storage'],
    cushion: ['cushion', 'pillow', 'throw'],
    pillow: ['pillow', 'cushion', 'throw'],
    shelf: ['shelf', 'bookcase'],
    drawer: ['drawer', 'cabinet', 'credenza', 'storage'],
    rug: ['rug', 'carpet', 'mat'],
    mirror: ['mirror', 'glass'],
    painting: ['painting', 'art', 'canvas'],
    ottoman: ['ottoman', 'pouf'],
    stool: ['stool', 'seat'],
    sideboard: ['sideboard', 'credenza'],
    console: ['console', 'sideboard']
  };

  const getFilteredRecommendations = () => {
    if (!result) return [];
    if (!selectedDetection) return result.recommendations.slice(0, 6);
    
    const label = selectedDetection.label.toLowerCase();
    const keywords = CATEGORY_MAP[label] || [label];
    
    const filtered = result.recommendations.filter(item => {
      const name = item.name.toLowerCase();
      return keywords.some(kw => name.includes(kw.toLowerCase()));
    });
    
    return filtered.slice(0, 6);
  };

  const getAmazonSearchUrl = () => {
    let query = ''
    if (selectedDetection) {
      query = `${result?.style || 'Modern'} ${selectedDetection.label} furniture`
    } else {
      query = `${result?.style || 'Modern'} style interior design furniture`
    }
    return `https://www.amazon.com/s?k=${encodeURIComponent(query)}`
  };

  const displayRecs = getFilteredRecommendations();

  const accentColor = result ? (STYLE_ACCENTS[result.style] ?? '#A8DADC') : '#A8DADC'

  // Update CSS --accent variable dynamically when style is detected
  React.useEffect(() => {
    const root = document.documentElement
    root.style.setProperty('--accent', accentColor)
    root.style.setProperty('--accent-dim', `${accentColor}1a`)
    root.style.setProperty('--accent-glow', `${accentColor}30`)
  }, [accentColor])

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPG, PNG, WEBP)')
      return
    }
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setSelectedDetection(null)
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [])

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleAnalyze = async () => {
    if (!imageFile) return
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('image', imageFile)
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.detail ?? `Server error ${response.status}`)
      }
      const data: AnalysisResult = await response.json()
      setResult(data)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      // Only prefix if it's actually a network error, otherwise show the clean server error message
      if (message === 'Failed to fetch' || message.includes('NetworkError')) {
        setError(`Network Error: Could not connect to backend at ${API_URL}`)
      } else {
        setError(message)
      }
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setImageFile(null)
    setImagePreview(null)
    setResult(null)
    setError(null)
    setSelectedDetection(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <div className="app-wrapper">
      <nav className="navbar">
        <div className="navbar-logo">
          <div className="logo-icon" />
          <span className="logo-text">SPAC3D</span>
        </div>
        <span className="navbar-tagline">
          AI Room Analysis &amp; Furniture Intelligence
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--text-muted)' }}>
          <span className="live-dot" />
          Connected
        </div>
      </nav>

      <main className="main-content">

        <div className="glass-panel">
          <p className="panel-title">Room Image</p>

          {!imagePreview ? (
            /* Drop Zone */
            <div
              id="upload-dropzone"
              className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={() => setIsDragOver(false)}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
              <div className="upload-icon" />
              <p className="upload-title">Drop your room photo here</p>
              <p className="upload-sub">
                or <span>click to browse</span>
              </p>
              <p className="upload-sub" style={{ fontSize: 12, marginTop: 4 }}>
                JPG · PNG · WEBP · max 20MB
              </p>
            </div>
          ) : (
            <ImageWithBboxes
              src={imagePreview}
              detections={result?.detections ?? []}
              imageSize={result?.image_size}
              accentColor={accentColor}
              selectedDetection={selectedDetection}
              onSelectDetection={setSelectedDetection}
            />
          )}

          {error && (
            <div className="error-banner">
              {error}
            </div>
          )}

          {imagePreview && (
            <>
              <button
                id="btn-analyze"
                className="btn-analyze"
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? (
                  <><div className="spinner" /> Analyzing Room...</>
                ) : (
                  <>Analyze Room</>
                )}
              </button>
              <button className="btn-reset" onClick={handleReset}>
                Clear &amp; Start Over
              </button>
            </>
          )}
        </div>

        <div className="glass-panel" style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 180px)' }}>
          <p className="panel-title">Analysis Results</p>

          {!result && !loading && (
            <div className="empty-state">
              <div className="empty-state-icon" />
              <p className="empty-state-text">
                Upload a room photo and click Analyze<br />to get style detection &amp; furniture recommendations
              </p>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div style={{ fontSize: 40, opacity: 0.3 }} />
              <p className="empty-state-text">
                Running ML pipeline...<br />
                <span style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6, display: 'block' }}>
                  YOLOv8 → ResNet50 → PCA → kNN → SVM
                </span>
              </p>
            </div>
          )}

          {result && (
            <div className="fade-in-up">
            <div className="style-result">
                <p className="style-label">Detected Room Style</p>
                <p className="style-name">{result.style}</p>
                <div className="confidence-row">
                  <span className="confidence-label">SVM Confidence</span>
                  <span className="confidence-value">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="confidence-bar-track">
                  <div
                    className="confidence-bar-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
              </div>

              {result.detections.length > 0 && (
                <div className="detections-section">
                  <p className="detections-header">
                    {result.detections.length} object{result.detections.length !== 1 ? 's' : ''} detected
                  </p>
                  <div className="detection-chips">
                    {result.detections.map((d, i) => (
                      <div className="detection-chip" key={i}>
                        <span className="chip-dot" />
                        {d.label}
                        <span style={{ opacity: 0.5, fontSize: 11 }}>
                          {(d.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.recommendations.length > 0 && (
                <div style={{ marginTop: 24, paddingTop: 16, borderTop: '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                    <p className="recommendations-title" style={{ margin: 0 }}>
                      {selectedDetection 
                        ? `Recommendations for '${selectedDetection.label}'` 
                        : 'Furniture Recommendations'}
                    </p>
                    {selectedDetection && (
                      <button 
                        className="btn-reset" 
                        style={{ padding: '4px 12px', minHeight: 'unset', fontSize: 11, margin: 0, opacity: 0.8 }}
                        onClick={() => setSelectedDetection(null)}
                      >
                        Clear Selection
                      </button>
                    )}
                  </div>

                  <div style={{ marginBottom: 20 }}>
                    <a 
                      href={getAmazonSearchUrl()} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '8px',
                        backgroundColor: '#FF9900',
                        color: '#111',
                        fontWeight: 600,
                        fontSize: '13px',
                        padding: '10px 18px',
                        borderRadius: '6px',
                        textDecoration: 'none',
                        transition: 'all 0.2s ease',
                        boxShadow: '0 4px 12px rgba(255, 153, 0, 0.2)'
                      }}
                      onMouseOver={(e) => {
                        e.currentTarget.style.backgroundColor = '#fca41c'
                        e.currentTarget.style.transform = 'translateY(-1px)'
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.backgroundColor = '#FF9900'
                        e.currentTarget.style.transform = 'translateY(0)'
                      }}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="9" cy="21" r="1"></circle><circle cx="20" cy="21" r="1"></circle>
                        <path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path>
                      </svg>
                      Shop {selectedDetection ? `'${selectedDetection.label}'` : 'this Style'} on Amazon
                    </a>
                  </div>
                  
                  {displayRecs.length === 0 ? (
                    <div className="empty-state" style={{ padding: '24px 0', minHeight: 'unset' }}>
                      <p className="empty-state-text" style={{ fontSize: 13, opacity: 0.6 }}>
                        No exact matches found for '{selectedDetection?.label}' in this room style.
                      </p>
                    </div>
                  ) : (
                    <div className="furniture-grid">
                      {displayRecs.map((item) => (
                        <FurnitureCard key={item.id} item={item} />
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────── */
/* Image with SVG bounding box overlay                                 */
/* ─────────────────────────────────────────────────────────────────── */
function ImageWithBboxes({
  src,
  detections,
  imageSize,
  accentColor,
  selectedDetection,
  onSelectDetection,
}: {
  src: string
  detections: Detection[]
  imageSize?: { width: number; height: number }
  accentColor: string
  selectedDetection?: Detection | null
  onSelectDetection?: (det: Detection) => void
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 })

  const onLoad = () => {
    if (imgRef.current) {
      setDisplaySize({
        w: imgRef.current.offsetWidth,
        h: imgRef.current.offsetHeight,
      })
    }
  }

  // Scale pixel coords from original image size to displayed size
  const scaleX = displaySize.w / ((imageSize?.width ?? displaySize.w) || 1)
  const scaleY = displaySize.h / ((imageSize?.height ?? displaySize.h) || 1)

  return (
    <div className="image-container">
      <img ref={imgRef} src={src} alt="Room" onLoad={onLoad} style={{ width: '100%' }} />

      {detections.length > 0 && displaySize.w > 0 && (
        <svg
          className="bbox-overlay"
          viewBox={`0 0 ${displaySize.w} ${displaySize.h}`}
          xmlns="http://www.w3.org/2000/svg"
        >
          {detections.map((det, i) => {
            const [x1, y1, x2, y2] = det.bbox
            const rx = x1 * scaleX
            const ry = y1 * scaleY
            const rw = (x2 - x1) * scaleX
            const rh = (y2 - y1) * scaleY
            
            const isSelected = selectedDetection && 
              selectedDetection.label === det.label && 
              selectedDetection.bbox[0] === det.bbox[0] &&
              selectedDetection.bbox[1] === det.bbox[1]

            return (
              <g 
                key={i} 
                onClick={() => onSelectDetection && onSelectDetection(det)}
                style={{ cursor: onSelectDetection ? 'pointer' : 'default', transition: 'all 0.2s ease', pointerEvents: 'all' }}
              >
                <rect
                  x={rx} y={ry} width={rw} height={rh}
                  fill={isSelected ? `${accentColor}25` : "transparent"}
                  stroke={accentColor}
                  strokeWidth={isSelected ? "3" : "2"}
                  strokeDasharray={isSelected ? "none" : "6 3"}
                  opacity={isSelected ? "1" : "0.7"}
                />
                <rect
                  x={rx} y={ry - 22} width={det.label.length * 8 + 20} height={20}
                  fill={accentColor}
                  rx="4"
                  opacity={isSelected ? "1" : "0.9"}
                />
                <text
                  x={rx + 8} y={ry - 7}
                  fill="#0a0a0f"
                  fontSize="11"
                  fontWeight="700"
                  fontFamily="Inter, sans-serif"
                  textAnchor="start"
                >
                  {det.label} {(det.confidence * 100).toFixed(0)}%
                </text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────── */
/* Furniture recommendation card                                        */
/* ─────────────────────────────────────────────────────────────────── */
function FurnitureCard({ item }: { item: FurnitureItem }) {
  const nameLower = item.name.toLowerCase()
  let keyword = 'furniture'
  const lookups = [
    'chair', 'sofa', 'couch', 'bed', 'table', 'plant', 'vase', 
    'lamp', 'clock', 'tv', 'shelf', 'drawer', 'rug', 'mirror', 
    'painting', 'cabinet', 'basket', 'console', 'ottoman', 'desk'
  ]
  for (const l of lookups) {
    if (nameLower.includes(l)) {
      keyword = l
      break
    }
  }

  let hash = 0
  for (let i = 0; i < item.id.length; i++) {
    hash = item.id.charCodeAt(i) + ((hash << 5) - hash)
  }
  const lockId = Math.abs(hash % 10000)

  const imgSrc = `https://loremflickr.com/400/300/interior,${keyword}?lock=${lockId}`

  const getProductSearchUrl = () => {
    const query = `${item.style} ${item.name} furniture`
    return `https://www.amazon.com/s?k=${encodeURIComponent(query)}`
  }

  return (
    <div className="furniture-card" style={{ display: 'flex', flexDirection: 'column' }}>
      <img
        src={imgSrc}
        alt={item.name}
        className="furniture-card-img"
      />
      <div className="furniture-card-body" style={{ display: 'flex', flexDirection: 'column', flexGrow: 1 }}>
        <p className="furniture-card-name">{item.name}</p>
        <div className="furniture-card-meta" style={{ marginBottom: '16px' }}>
          <span className="furniture-style-tag">{item.style}</span>
          <div className="furniture-similarity">
            <div className="similarity-bar">
              <div
                className="similarity-fill"
                style={{ width: `${item.similarity_score * 100}%` }}
              />
            </div>
            <span className="similarity-score">
              {(item.similarity_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        
        <div style={{ marginTop: 'auto' }}>
          <a 
            href={getProductSearchUrl()}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              padding: '8px 12px',
              backgroundColor: '#f3f4f6',
              color: '#374151',
              fontSize: '12px',
              fontWeight: 600,
              textDecoration: 'none',
              borderRadius: '6px',
              border: '1px solid #e5e7eb',
              transition: 'all 0.2s ease',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = '#e5e7eb'
              e.currentTarget.style.borderColor = '#d1d5db'
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = '#f3f4f6'
              e.currentTarget.style.borderColor = '#e5e7eb'
            }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="9" cy="21" r="1"></circle><circle cx="20" cy="21" r="1"></circle>
              <path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path>
            </svg>
            Shop this item
          </a>
        </div>
      </div>
    </div>
  )
}

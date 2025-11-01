// API 設置 (請確保您的 EC2 IP 和 port 是正確的)
const API_URL = "http://52.194.220.86:8080/predict";

// 標籤映射
const SENTIMENT_LABELS = {
    '0': '正面',
    '1': '中立',
    '2': '負面',
    '3': '無關',
};

// 顏色映射
const COLOR_MAP = {
    '0': '#4E79A7', // Red
    '1': '#F28E2B', // Yellow-ish
    '2': '#E15759', // Grey
    '3': '#76B7B2', // Green
};

// -----------------------------------------------------------------
// 1. 爬蟲邏輯 (之前在 content.js 中)
// -----------------------------------------------------------------
/**
 * 此函數將被注入到 Threads 頁面中執行。
 * 它不能存取 popup.js 中的任何變數，這是一個獨立的世界。
 */
function scrapeCommentsFromPage() {
    // 這是您之前測試成功的 CSS 選擇器
    const CSS_SELECTOR_FOR_CANDIDATES = 'span[dir="auto"] > span';
    const MIN_LENGTH = 2; // 最小評論長度 (過濾雜訊)

    try {
        const candidates = document.querySelectorAll(CSS_SELECTOR_FOR_CANDIDATES);
        const comments = [];

        candidates.forEach(node => {
            // 關鍵過濾邏輯：檢查父層是否為 <a> 標籤
            const parentElement = node.parentElement;
            if (parentElement && parentElement.closest('a')) {
                // 這是使用者 ID，跳過
                return;
            }

            const text = node.textContent.trim();
            if (text.length > MIN_LENGTH) {
                comments.push(text);
            }
        });
        
        // 🌟 直接返回結果 (這將成為一個 Promise)
        return comments;

    } catch (error) {
        return { error: error.message };
    }
}

// -----------------------------------------------------------------
// 2. UI 和 API 邏輯 (popup.js)
// -----------------------------------------------------------------

function setStatus(message, isError = false) {
    const statusEl = document.getElementById('status');
    statusEl.innerText = message;
    statusEl.className = isError ? 'status-message error' : 'status-message';
}

function displayResults(sentimentCounts, totalCount, comments) {
    // 顯示統計摘要
    document.getElementById('totalCount').innerText = totalCount;
    
    // 顯示情感細分
    const breakdownEl = document.getElementById('sentimentBreakdown');
    breakdownEl.innerHTML = '<h4>情感分類細項</h4>';

    let chartDataLabels = [];
    let chartDataValues = [];
    let chartDataColors = [];

    // 填充細分區塊和圖表數據
    Object.keys(sentimentCounts).forEach(label => {
        const count = sentimentCounts[label];
        const percentage = ((count / totalCount) * 100).toFixed(1);
        const description = SENTIMENT_LABELS[label] || `未知標籤 (${label})`;
        const color = COLOR_MAP[label] || '#cccccc';

        breakdownEl.innerHTML += `
            <p style="color: #333; display: flex; align-items: center; margin-bottom: 8px;">
                <span style="width: 12px; height: 12px; background-color: ${color}; border-radius: 3px; margin-right: 8px; flex-shrink: 0;"></span>
                <span style="flex-grow: 1;">
                    ${description}: <strong>${count} 則</strong> (${percentage}%)
                </span>
            </p>`;

        chartDataLabels.push(description);
        chartDataValues.push(count);
        chartDataColors.push(color);
    });

    // 繪製圓餅圖
    const ctx = document.getElementById('sentimentChart').getContext('2d');
    if (window.sentimentChartInstance) {
        window.sentimentChartInstance.destroy();
    }
    window.sentimentChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: chartDataLabels,
            datasets: [{
                data: chartDataValues,
                backgroundColor: chartDataColors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { position: 'top' }, title: { display: true, text: '情感分佈圖' } }
        }
    });

    // 顯示結果容器
    document.getElementById('resultsContainer').style.display = 'block';
    
    // 顯示評論列表 (用於除錯)
    const ul = document.getElementById('commentUl');
    ul.innerHTML = comments.map(c => `<li>${c}</li>`).join('');
    document.getElementById('readCount').innerText = totalCount;
    document.getElementById('commentList').style.display = 'block';
}

async function fetchPredictions(comments) {
    setStatus("🚀 正在分析情感，請稍候...");
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ comments: comments })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API 請求失敗: ${response.status} - ${errorText}`);
        }

        const predictions = await response.json();
        
        let sentimentCounts = {};
        predictions.forEach(item => {
            const rawSentiment = item.sentiment.toString();
            const sentiment = rawSentiment.replace("LABEL_", "");
            sentimentCounts[sentiment] = (sentimentCounts[sentiment] || 0) + 1;
        });

        setStatus("✅ 分析完成！");
        displayResults(sentimentCounts, comments.length, comments);

    } catch (error) {
        console.error("API Error:", error);
        setStatus(`💔 分析失敗: ${error.message}`, true);
    }
}

// -----------------------------------------------------------------
// 3. 主事件監聽器 (更新)
// -----------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    const analyzeButton = document.getElementById('analyzeButton');
    
    // 🌟 (已移除 'chrome.runtime.onMessage' 監聽器)

    analyzeButton.addEventListener('click', async () => {
        setStatus("🔍 正在讀取 Threads 頁面評論...");
        
        let activeTab;
        try {
            const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
            if (tabs.length === 0) {
                throw new Error("無法獲取當前頁面標籤。");
            }
            activeTab = tabs[0];
            
            // 🌟 核心修改：注入 'scrapeCommentsFromPage' 函數
            const injectionResults = await chrome.scripting.executeScript({
                target: { tabId: activeTab.id },
                func: scrapeCommentsFromPage // 直接注入函數
            });

            // 執行結果會被包裹在一個陣列中
            const result = injectionResults[0].result;

            if (result.error) {
                throw new Error(result.error);
            }

            const comments = result;
            if (comments.length === 0) {
                setStatus("🚫 頁面上找不到任何評論或格式不正確。", true);
                return;
            }
            
            setStatus(`✅ 成功讀取 ${comments.length} 則評論。`);
            
            // 收到評論後，立即呼叫 API 進行預測
            await fetchPredictions(comments);

        } catch (e) {
            console.error("執行腳本失敗:", e);
            setStatus(`無法執行內容腳本：${e.message}`, true);
        }
    });
});


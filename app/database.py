<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://static.line-scdn.net/liff/edge/2/sdk.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" />
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js"></script>
    <title>ผลการคำนวณคะแนน</title>
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(135deg, #0288d1 0%, #01579b 100%);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 10px;
            min-height: 100vh;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 0 10px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 20px;
            padding: 20px 0;
        }

        .header h1 {
            margin: 0;
            font-size: 1.8rem;
            font-weight: 600;
        }

        .header p {
            margin: 8px 0 0 0;
            opacity: 0.9;
            font-size: 1rem;
        }

        #profile {
            text-align: center;
            margin-bottom: 20px;
            color: white;
        }

        #profile img {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            margin-bottom: 8px;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }

        #profile h2 {
            margin: 8px 0 4px 0;
            font-size: 1.2rem;
        }

        #profile .user-id {
            font-size: 0.9rem;
            opacity: 0.8;
            margin: 0;
        }

        .loading {
            text-align: center;
            color: white;
            padding: 40px;
            font-size: 1.1rem;
        }

        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin: 0 auto 16px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .debug-info {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 20px;
            color: white;
            font-size: 0.85rem;
        }

        .debug-info pre {
            background: rgba(0,0,0,0.3);
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 8px 0;
        }

        .user-summary {
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            color: white;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }

        .summary-item {
            text-align: center;
            background: rgba(255,255,255,0.1);
            padding: 16px;
            border-radius: 12px;
        }

        .summary-label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-bottom: 8px;
        }

        .summary-value {
            font-size: 1.4rem;
            font-weight: bold;
        }

        .university-card {
            background: white;
            border-radius: 16px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .card-header {
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: white;
        }

        .card-header.calculated {
            background: linear-gradient(135deg, #4caf50, #388e3c);
        }

        .card-header.gpax-insufficient {
            background: linear-gradient(135deg, #ff9800, #f57c00);
        }

        .card-header.not-found {
            background: linear-gradient(135deg, #f44336, #d32f2f);
        }

        .card-header.incomplete {
            background: linear-gradient(135deg, #9e9e9e, #616161);
        }

        .card-header.error {
            background: linear-gradient(135deg, #e91e63, #c2185b);
        }

        .card-header h3 {
            margin: 0;
            font-size: 1.1rem;
            font-weight: 600;
        }

        .university-info {
            font-size: 0.9rem;
            opacity: 0.9;
            margin-top: 4px;
        }

        .card-number {
            background: rgba(255,255,255,0.2);
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9rem;
        }

        .card-body {
            padding: 20px;
        }

        .score-display {
            text-align: center;
            padding: 20px;
            background: #f8faff;
            border-radius: 12px;
            margin-bottom: 16px;
        }

        .score-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #01579b;
            margin-bottom: 8px;
        }

        .score-label {
            font-size: 0.9rem;
            color: #666;
        }

        .status-message {
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.9rem;
            margin-bottom: 16px;
        }

        .status-message.success {
            background: #e8f5e8;
            color: #388e3c;
            border: 1px solid #c8e6c9;
        }

        .status-message.warning {
            background: #fff8e1;
            color: #f57c00;
            border: 1px solid #ffecb3;
        }

        .status-message.error {
            background: #ffebee;
            color: #d32f2f;
            border: 1px solid #ffcdd2;
        }

        .status-message.info {
            background: #e3f2fd;
            color: #0288d1;
            border: 1px solid #bbdefb;
        }

        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #666;
            font-style: italic;
            background: #f8faff;
            border-radius: 12px;
            border: 2px dashed #e3f2fd;
        }

        .error-state {
            text-align: center;
            padding: 40px 20px;
            color: white;
            background: rgba(244, 67, 54, 0.1);
            border-radius: 12px;
            border: 2px dashed rgba(244, 67, 54, 0.3);
        }

        .retry-btn, .debug-btn {
            background: #0288d1;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 0.9rem;
            cursor: pointer;
            margin: 8px;
        }

        .retry-btn:hover, .debug-btn:hover {
            background: #0277bd;
        }

        .footer {
            text-align: center;
            color: white;
            opacity: 0.8;
            margin-top: 30px;
            padding: 20px 0;
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            .container {
                padding: 0 5px;
            }
            
            .header h1 {
                font-size: 1.5rem;
            }
            
            .card-body {
                padding: 16px;
            }
            
            .score-value {
                font-size: 2rem;
            }

            .summary-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div id="profile">
                <img id="picture" src="" alt="Profile Picture" style="display: none;">
                <h2 id="name">Loading...</h2>
                <p id="userId" class="user-id"></p>
            </div>
            <h1>ผลการคำนวณคะแนน</h1>
            <p>ผลการประเมินโอกาสเข้าศึกษาต่อมหาวิทยาลัย</p>
        </div>

        <div id="loadingSection" class="loading">
            <div class="loading-spinner"></div>
            <p>กำลังคำนวณคะแนน...</p>
        </div>

        <div id="errorSection" class="error-state" style="display: none;">
            <h3>เกิดข้อผิดพลาด</h3>
            <p id="errorMessage">ไม่สามารถโหลดข้อมูลได้</p>
            <button class="retry-btn" onclick="loadScoreData()">ลองอีกครั้ง</button>
            <button class="debug-btn" onclick="loadDebugData()">ดูข้อมูลดีบัก</button>
        </div>

        <div id="debugSection" class="debug-info" style="display: none;">
            <h3>ข้อมูลสำหรับดีบัก</h3>
            <div id="debugContent"></div>
            <button class="retry-btn" onclick="hideDebug()">ซ่อน</button>
        </div>

        <div id="resultsSection" style="display: none;">
            <!-- User Summary will be inserted here -->
            <div id="userSummary"></div>
            
            <!-- University cards will be inserted here -->
            <div id="universityResults"></div>
        </div>

        <div class="footer">
            <p>&copy; 2024 University Score Calculator. Last updated: <span id="lastUpdated"></span></p>
        </div>
    </div>

    <script>
        let currentUserId = null;
        let debugMode = false;

        window.onload = async function () {
            try {
                await liff.init({ liffId: "2007611527-z7ZNLXOk" });
                
                if (liff.isLoggedIn()) {
                    const profile = await liff.getProfile();
                    currentUserId = profile.userId;
                    
                    document.getElementById("name").innerText = profile.displayName;
                    document.getElementById("userId").innerText = `ID: ${profile.userId}`;
                    document.getElementById("picture").src = profile.pictureUrl;
                    document.getElementById("picture").style.display = "block";
                    
                    console.log("User ID:", currentUserId);
                    
                    // Load score data
                    await loadScoreData();
                } else {
                    liff.login();
                }
            } catch (error) {
                console.error('LIFF initialization failed:', error);
                showError("ไม่สามารถเชื่อมต่อกับ LINE ได้: " + error.message);
            }
        };

        async function loadScoreData() {
            if (!currentUserId) {
                showError("ไม่พบข้อมูลผู้ใช้");
                return;
            }

            try {
                showLoading();
                console.log("Loading score data for user:", currentUserId);
                
                const response = await fetch("https://varee.onrender.com/api/calculate_scores", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ userId: currentUserId })
                });

                console.log("Response status:", response.status);
                console.log("Response headers:", response.headers);

                if (!response.ok) {
                    const errorText = await response.text();
                    console.error("Response error:", errorText);
                    throw new Error(`HTTP ${response.status}: ${errorText}`);
                }

                const data = await response.json();
                console.log("Received data:", data);
                
                if (data.error) {
                    throw new Error(data.error);
                }

                displayResults(data);
                
            } catch (error) {
                console.error('Error loading score data:', error);
                showError(error.message || "ไม่สามารถโหลดข้อมูลคะแนนได้");
            }
        }

        async function loadDebugData() {
            if (!currentUserId) {
                showError("ไม่พบข้อมูลผู้ใช้");
                return;
            }

            try {
                console.log("Loading debug data for user:", currentUserId);
                
                const response = await fetch(`https://varee.onrender.com/api/debug/user/${currentUserId}`, {
                    method: "GET",
                    headers: {
                        "Content-Type": "application/json"
                    }
                });

                console.log("Debug response status:", response.status);

                if (!response.ok) {
                    const errorText = await response.text();
                    console.error("Debug response error:", errorText);
                    throw new Error(`HTTP ${response.status}: ${errorText}`);
                }

                const data = await response.json();
                console.log("Debug data:", data);
                
                displayDebugInfo(data);
                
            } catch (error) {
                console.error('Error loading debug data:', error);
                showError("ไม่สามารถโหลดข้อมูลดีบัก: " + error.message);
            }
        }

        function displayDebugInfo(data) {
            const debugContent = document.getElementById("debugContent");
            debugContent.innerHTML = `
                <h4>ข้อมูลผู้ใช้:</h4>
                <pre>${JSON.stringify(data, null, 2)}</pre>
            `;
            
            document.getElementById("debugSection").style.display = "block";
        }

        function hideDebug() {
            document.getElementById("debugSection").style.display = "none";
        }

        function showLoading() {
            document.getElementById("loadingSection").style.display = "block";
            document.getElementById("errorSection").style.display = "none";
            document.getElementById("resultsSection").style.display = "none";
            document.getElementById("debugSection").style.display = "none";
        }

        function showError(message) {
            document.getElementById("loadingSection").style.display = "none";
            document.getElementById("errorSection").style.display = "block";
            document.getElementById("resultsSection").style.display = "none";
            document.getElementById("errorMessage").innerText = message;
            console.error("Error:", message);
        }

        function displayResults(data) {
            document.getElementById("loadingSection").style.display = "none";
            document.getElementById("errorSection").style.display = "none";
            document.getElementById("resultsSection").style.display = "block";
            document.getElementById("lastUpdated").innerText = new Date().toLocaleDateString('th-TH');

            console.log("Displaying results:", data);

            // Display user summary
            displayUserSummary(data);
            
            // Display university results
            displayUniversityResults(data.results || []);
        }

        function displayUserSummary(data) {
            const summary = data.summary || {};
            const debug = data.debug || {};
            
            const summaryHtml = `
                <div class="user-summary">
                    <h3>สรุปผลการประเมิน</h3>
                    <div class="summary-grid">
                        <div class="summary-item">
                            <div class="summary-label">ชื่อผู้ใช้</div>
                            <div class="summary-value">${debug.user_name || 'ไม่ระบุ'}</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">GPAX ของคุณ</div>
                            <div class="summary-value">${debug.user_gpax || 'ไม่ระบุ'}</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">จำนวนการเลือก</div>
                            <div class="summary-value">${summary.total_selections || 0}</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">คำนวณได้</div>
                            <div class="summary-value">${summary.calculated_scores || 0}</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">คะแนนสูงสุด</div>
                            <div class="summary-value">${summary.highest_score ? summary.highest_score.toFixed(2) : '0'}</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">Cache Status</div>
                            <div class="summary-value">${debug.cache_valid ? '✓' : '✗'}</div>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById("userSummary").innerHTML = summaryHtml;
        }

        function displayUniversityResults(results) {
            const container = document.getElementById("universityResults");
            container.innerHTML = "";

            console.log("Displaying university results:", results);

            if (results.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <h3>ยังไม่มีข้อมูลการเลือกมหาวิทยาลัย</h3>
                        <p>กรุณาเลือกมหาวิทยาลัยที่ต้องการสมัครก่อน</p>
                    </div>
                `;
                return;
            }

            results.forEach((result, index) => {
                const card = createUniversityCard(result, index + 1);
                container.appendChild(card);
            });
        }

        function createUniversityCard(result, cardNumber) {
            const div = document.createElement('div');
            div.className = 'university-card';
            
            console.log("Creating card for result:", result);
            
            const university = result.university || 'ไม่ระบุมหาวิทยาลัย';
            const faculty = result.faculty || 'ไม่ระบุคณะ';
            const field = result.field || 'ไม่ระบุสาขา';
            const status = result.status || 'unknown';
            const score = result.score;
            const message = result.message || 'ไม่มีข้อความ';
            
            let headerClass = 'calculated';
            let statusClass = 'success';
            
            switch (status) {
                case 'calculated':
                    headerClass = 'calculated';
                    statusClass = 'success';
                    break;
                case 'gpax_insufficient':
                    headerClass = 'gpax-insufficient';
                    statusClass = 'warning';
                    break;
                case 'not_found':
                    headerClass = 'not-found';
                    statusClass = 'error';
                    break;
                case 'incomplete':
                    headerClass = 'incomplete';
                    statusClass = 'info';
                    break;
                case 'error':
                    headerClass = 'error';
                    statusClass = 'error';
                    break;
            }

            div.innerHTML = `
                <div class="card-header ${headerClass}">
                    <div>
                        <h3>${university}</h3>
                        <div class="university-info">
                            ${faculty} - ${field}
                        </div>
                    </div>
                    <div class="card-number">${cardNumber}</div>
                </div>
                <div class="card-body">
                    <div class="status-message ${statusClass}">
                        ${message}
                    </div>
                    
                    ${status === 'calculated' && score !== null ? `
                        <div class="score-display">
                            <div class="score-value">${score.toFixed(2)}</div>
                            <div class="score-label">คะแนนรวม</div>
                        </div>
                    ` : ''}
                    
                    ${status === 'incomplete' ? `
                        <div class="empty-state">
                            <p>ข้อมูลการเลือกไม่สมบูรณ์</p>
                        </div>
                    ` : ''}
                </div>
            `;
            
            return div;
        }
    </script>
</body>
</html>

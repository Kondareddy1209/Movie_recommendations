    <%@ page contentType="text/html;charset=UTF-8" language="java" %>
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Movie Recommendation System</title>
        <link rel="stylesheet" href="${pageContext.request.contextPath}/static/style.css">
    </head>
    <body>
        <div class="container">
            <h1>Movie Recommendation System</h1>
            <form method="POST" action="/recommend">
                <label for="user_id">Enter User ID:</label>
                <input type="text" id="user_id" name="user_id" required>
                <button type="submit">Get Recommendations</button>
            </form>
            
            <c:if test="${not empty error}">
                <p class="error">${error}</p>
            </c:if>
            
            <c:if test="${not empty user_id}">
                <h2>Recommendations for User ${user_id}:</h2>
                <ul class="recommendations">
                    <c:forEach var="rec" items="${recommendations}">
                        <li>${rec}</li>
                    </c:forEach>
                </ul>
            </c:if>
        </div>
    </body>
    </html>

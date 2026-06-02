from src.rag.api.app import create_app

if __name__ == "__main__":
    app = create_app(autoload=True)
    app.run(debug=True)

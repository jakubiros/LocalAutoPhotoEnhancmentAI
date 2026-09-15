import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from config import PathInfo

CURRENT_DIR = Path(__file__).resolve().parent

paths=PathInfo()
INBOX_DIR=Path(paths.input_dir)
OUTBOX_DIR=Path(paths.output_dir)

INBOX_DIR.mkdir(parents=True, exist_ok=True)
OUTBOX_DIR.mkdir(parents=True, exist_ok=True)

def create_app(process_fn):
    app = FastAPI(title='Edge Photo Enhancement AI')

    def run_pipeline_task(file_path: Path):
        try:
            process_fn(str(file_path), str(OUTBOX_DIR))
            print(f'{file_path.name} processed successfully')
        except Exception as e:
            print(f'[Error] {file_path.name}: {e}')
        finally:
            if file_path.exists():
                file_path.unlink()

    @app.post('/api/upload')
    async def upload_photos(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
        for f in files:
            dest=INBOX_DIR / f.filename
            with open(dest, 'wb') as buffer:
                shutil.copyfileobj(f.file, buffer)
                background_tasks.add_task(run_pipeline_task, dest)
        return {'status': 'queued', 'count': len(files)}

    @app.get('/api/results')
    async def get_results():
        files=[f.name for f in OUTBOX_DIR.glob("*_AI_enhanced.*")]
        return {'files': sorted(files, reverse=True)}

    @app.get('/api/download/{filename}')
    async def download_file(filename: str):
        target=OUTBOX_DIR / filename
        if target.exists():
            return FileResponse(target, media_type='image/jpeg', filename=filename)
        return {'error': 'File not found'}, 404

    @app.get('/api/download-all-zip')
    async def download_all_zip():
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in OUTBOX_DIR.glob('*.jpg'):
                zip_file.write(file_path, arcname=file_path.name)
        zip_buffer.seek(0)
        return StreamingResponse(zip_buffer, media_type='application/zip', headers={'Content-Disposition': 'attachment; filename=enhanced_photos.zip'})

    @app.get("/", response_class=HTMLResponse)
    async def home_ui():
        html_path = CURRENT_DIR / "index.html"
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"<h1>Critical Error</h1><p>Interface file not found: {html_path}</p>"

    return app
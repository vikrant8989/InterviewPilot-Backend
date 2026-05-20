from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.interview_service import InterviewService
from app.core.security import verify_jwt
from app.ws.session_manager import session_manager

router = APIRouter()
service = InterviewService()


@router.websocket("/interview")
async def interview_socket(websocket: WebSocket):
    await websocket.accept()

    session_id = websocket.query_params.get("sessionId")
    token = websocket.query_params.get("token")
    if not session_id:
        await websocket.send_json({"event": "error", "payload": {"code": "MISSING_SESSION_ID", "message": "sessionId is required"}})
        await websocket.close()
        return
    if not token:
        await websocket.send_json({"event": "error", "payload": {"code": "MISSING_TOKEN", "message": "token query param is required"}})
        await websocket.close()
        return
    auth = verify_jwt(token)
    user_id = auth["user_id"]

    await session_manager.connect(session_id=session_id, user_id=user_id, websocket=websocket)

    try:
        while True:
            msg = await websocket.receive_json()
            event = msg.get("event")
            payload = msg.get("payload", {})

            if event == "join_session":
                await service.on_join(websocket=websocket, session_id=session_id, user_id=user_id, payload=payload)

            elif event == "text_answer_submitted":
                await service.on_text_answer(websocket=websocket, session_id=session_id, user_id=user_id, payload=payload)

            elif event == "answer_chunk_uploaded":
                await service.on_audio_chunk_uploaded(websocket=websocket, session_id=session_id, user_id=user_id, payload=payload)

            elif event == "client_proctor_event":
                await service.on_proctor_event(websocket=websocket, session_id=session_id, user_id=user_id, payload=payload)

            else:
                await websocket.send_json({"event": "error", "payload": {"code": "UNKNOWN_EVENT", "message": f"Unknown event: {event}"}})

    except WebSocketDisconnect:
        await service.on_disconnect(session_id=session_id)
        await session_manager.disconnect(session_id=session_id, user_id=user_id, websocket=websocket)
    except Exception:
        pass


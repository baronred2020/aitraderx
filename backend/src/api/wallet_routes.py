from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.database_models import VirtualWallet, WalletTransaction, User
from config.database_config import get_db
from config.auth_config import get_current_user
from datetime import datetime
from typing import List

router = APIRouter()

@router.get("/wallet")
def get_wallet(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """Obtiene el balance y transacciones del wallet virtual"""
    try:
        username = current_user.get("username")
        if not username:
            raise HTTPException(status_code=401, detail="Usuario no autenticado")
        
        # Obtener usuario de la base de datos
        user = db.query(User).filter_by(username=username).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        wallet = db.query(VirtualWallet).filter_by(user_id=user.user_id).first()
        if not wallet:
            # Crear wallet inicial si no existe
            wallet = VirtualWallet(user_id=user.user_id, balance=10000)
            db.add(wallet)
            db.commit()
            db.refresh(wallet)
        
        transactions = db.query(WalletTransaction).filter_by(wallet_id=wallet.id).order_by(WalletTransaction.created_at.desc()).all()
        
        return {
            "balance": float(wallet.balance),
            "transactions": [
                {
                    "id": t.id,
                    "type": t.type,
                    "amount": float(t.amount),
                    "description": t.description,
                    "created_at": t.created_at.isoformat()
                } for t in transactions
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.post("/wallet/recharge")
def recharge_wallet(amount: float, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """Recarga el wallet virtual"""
    try:
        username = current_user.get("username")
        if not username:
            raise HTTPException(status_code=401, detail="Usuario no autenticado")
        
        # Obtener usuario de la base de datos
        user = db.query(User).filter_by(username=username).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        wallet = db.query(VirtualWallet).filter_by(user_id=user.user_id).first()
        if not wallet:
            wallet = VirtualWallet(user_id=user.user_id, balance=0)
            db.add(wallet)
            db.commit()
            db.refresh(wallet)
        
        if float(wallet.balance) > 0:
            raise HTTPException(status_code=400, detail="Solo puedes recargar si tu saldo es 0.")
        
        if amount <= 0 or amount > 10000:
            raise HTTPException(status_code=400, detail="El monto de recarga debe ser mayor a 0 y máximo 10,000.")
        
        wallet.balance = amount
        db.add(WalletTransaction(
            wallet_id=wallet.id, 
            type="recharge", 
            amount=amount, 
            description="Recarga de saldo", 
            created_at=datetime.utcnow()
        ))
        db.commit()
        
        return {"balance": float(wallet.balance)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.post("/wallet/trade")
def trade_wallet(amount: float, description: str = "", db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """Realiza una operación de trading (descuenta saldo)"""
    try:
        username = current_user.get("username")
        if not username:
            raise HTTPException(status_code=401, detail="Usuario no autenticado")
        
        # Obtener usuario de la base de datos
        user = db.query(User).filter_by(username=username).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        wallet = db.query(VirtualWallet).filter_by(user_id=user.user_id).first()
        if not wallet:
            raise HTTPException(status_code=400, detail="No tienes wallet virtual.")
        
        if amount <= 0:
            raise HTTPException(status_code=400, detail="El monto debe ser mayor a 0.")
        
        if float(wallet.balance) < amount:
            raise HTTPException(status_code=400, detail="Saldo insuficiente.")
        
        wallet.balance = float(wallet.balance) - amount
        db.add(WalletTransaction(
            wallet_id=wallet.id, 
            type="trade", 
            amount=-amount, 
            description=description, 
            created_at=datetime.utcnow()
        ))
        db.commit()
        
        return {"balance": float(wallet.balance)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/wallet/transactions")
def get_wallet_transactions(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    """Obtiene el historial de transacciones del wallet"""
    try:
        username = current_user.get("username")
        if not username:
            raise HTTPException(status_code=401, detail="Usuario no autenticado")
        
        # Obtener usuario de la base de datos
        user = db.query(User).filter_by(username=username).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        wallet = db.query(VirtualWallet).filter_by(user_id=user.user_id).first()
        if not wallet:
            raise HTTPException(status_code=400, detail="No tienes wallet virtual.")
        
        transactions = db.query(WalletTransaction).filter_by(wallet_id=wallet.id).order_by(WalletTransaction.created_at.desc()).all()
        
        return [
            {
                "id": t.id,
                "type": t.type,
                "amount": float(t.amount),
                "description": t.description,
                "created_at": t.created_at.isoformat()
            } for t in transactions
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}") 